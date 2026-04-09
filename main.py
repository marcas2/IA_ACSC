"""
main.py
API principal de ValvIA — Sistema de Detección de Valvulopatías

Endpoints:
  POST /api/v1/train/full          → SERVICIO 1: Entrenamiento con datos locales
  POST /api/v1/train/sample        → SERVICIO 2: Nueva muestra + entrenamiento incremental
  POST /api/v1/diagnose            → SERVICIO 3: Diagnóstico de valvulopatía
  GET  /api/v1/metrics             → Evaluación del modelo (accuracy, F1, ROC-AUC…)
  GET  /api/v1/metrics/history     → Historial de métricas
  GET  /api/v1/model/status        → Estado actual del modelo
  GET  /health                     → Health check
"""
import json
import logging
import math
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Agregar raíz al path (necesario cuando se corre desde /home/claude/valvulopatias)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import API_HOST, API_PORT, API_TITLE, API_VERSION
from models.valvulopatia_model import get_model, reload_model
from reentrena.diagnosis_service import run_diagnosis
from reentrena.metrics_service import get_metrics_history, run_evaluation
from services.new_sample_service import process_new_training_sample
from services.train_service import run_full_training

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("valvia.api")


def _sanitize_json_payload(value):
    """Asegura compatibilidad JSON estricta: convierte NaN/Infinity a None."""
    if isinstance(value, dict):
        return {str(k): _sanitize_json_payload(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_sanitize_json_payload(v) for v in value]

    if isinstance(value, tuple):
        return [_sanitize_json_payload(v) for v in value]

    if isinstance(value, np.ndarray):
        return [_sanitize_json_payload(v) for v in value.tolist()]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        fval = float(value)
        return fval if math.isfinite(fval) else None

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🫀 ValvIA iniciando…")
    reload_model()           # Intenta cargar modelo pre-entrenado
    model = get_model()
    if model.is_ready:
        logger.info("✅ Modelo cargado correctamente desde disco.")
    else:
        logger.warning(
            "⚠️  Modelo no encontrado. Ejecute POST /api/v1/train/full para entrenar."
        )
    yield
    logger.info("ValvIA detenido.")


# ── Aplicación ────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Sistema de Inteligencia Artificial para detección de valvulopatías "
        "a partir de fonocardiogramas (sonidos cardiacos WAV) y metadatos clínicos."
    ),
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Sistema"])
async def health_check():
    model = get_model()
    return {
        "status": "ok",
        "modelo_listo": model.is_ready,
        "muestras_entrenadas": model._n_trained
    }


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/model/status", tags=["Modelo"])
async def model_status():
    model = get_model()
    return {
        "entrenado": model.is_ready,
        "muestras_entrenadas": model._n_trained,
        "dimension_features": model._feature_dim,
        "tipo_modelo": type(model.model).__name__ if model.model else None
    }


# ─────────────────────────────────────────────────────────────────────────────
# SERVICIO 1 — Entrenamiento completo desde servidor local
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/train/full", tags=["Entrenamiento"])
async def train_full(background_tasks: BackgroundTasks):
    """
    **SERVICIO 1** — Descarga todos los audios y JSON del servidor local
    (172.16.10.200:5002) y entrena el modelo desde cero.
    
    Este proceso puede tomar varios minutos dependiendo del volumen de datos.
    Retorna métricas de entrenamiento y validación cruzada.
    """
    try:
        result = run_full_training()
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Error en entrenamiento completo")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# SERVICIO 2 — Nueva muestra para entrenamiento incremental
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/train/sample", tags=["Entrenamiento"])
async def train_new_sample(
    json_metadata: str   = Form(..., description="JSON clínico como string"),
    audio_principal: UploadFile = File(..., description="Audio principal WAV"),
    audio_ecg:   Optional[UploadFile] = File(None, description="Audio ECG WAV"),
    audio_ecg_1: Optional[UploadFile] = File(None, description="Audio ECG_1 WAV"),
    audio_ecg_2: Optional[UploadFile] = File(None, description="Audio ECG_2 WAV"),
):
    """
    **SERVICIO 2** — Recibe hasta 4 archivos WAV y un JSON clínico.
    
    - Almacena cada archivo en su carpeta correspondiente del servidor local
    - Actualiza el modelo con entrenamiento incremental (partial_fit)
    
    El JSON **debe** incluir `diagnostico.estado` (ej: "Normal" o "Estenosis aórtica")
    para poder usarlo como etiqueta de entrenamiento.
    
    Carpetas en el servidor local:
    - audio_principal → Audios/
    - audio_ecg       → ECG/
    - audio_ecg_1     → ECG_1/
    - audio_ecg_2     → ECG_2/
    - JSON            → audios-json/
    """
    # Parsear JSON
    try:
        metadata = json.loads(json_metadata)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON inválido: {e}")

    # Leer archivos
    wav_main_bytes = await audio_principal.read()
    wav_ecg_bytes  = await audio_ecg.read()  if audio_ecg   else None
    wav_ecg1_bytes = await audio_ecg_1.read() if audio_ecg_1 else None
    wav_ecg2_bytes = await audio_ecg_2.read() if audio_ecg_2 else None

    # Nombres de archivo
    ecg_name  = audio_ecg.filename   if audio_ecg   else None
    ecg1_name = audio_ecg_1.filename if audio_ecg_1 else None
    ecg2_name = audio_ecg_2.filename if audio_ecg_2 else None

    try:
        result = process_new_training_sample(
            json_data=metadata,
            wav_principal_bytes=wav_main_bytes,
            wav_principal_name=audio_principal.filename,
            wav_ecg_bytes=wav_ecg_bytes,   wav_ecg_name=ecg_name,
            wav_ecg1_bytes=wav_ecg1_bytes, wav_ecg1_name=ecg1_name,
            wav_ecg2_bytes=wav_ecg2_bytes, wav_ecg2_name=ecg2_name,
        )
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Error procesando nueva muestra")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# SERVICIO 3 — Diagnóstico de valvulopatía
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/diagnose", tags=["Diagnóstico"])
async def diagnose(
    json_metadata: str = Form(..., description="JSON clínico del paciente"),
    audio:         UploadFile = File(..., description="Audio cardiaco WAV"),
):
    """
    **SERVICIO 3** — Análisis diagnóstico de valvulopatía.
    
    Recibe:
    - `json_metadata`: JSON con datos del paciente (edad, género, peso, foco de auscultación…)
    - `audio`: Archivo WAV del sonido cardiaco
    
    Retorna:
    - Diagnóstico: Normal / Anomalía
    - Probabilidad de valvulopatía (0–1)
    - Nivel de confianza (Alta / Media / Baja)
    - Recomendación clínica
    - Estado del almacenamiento en servidor local
    
    El audio se almacena automáticamente en el servidor local (172.16.10.200:5002).
    """
    try:
        metadata = json.loads(json_metadata)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON inválido: {e}")

    if not audio.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos WAV.")

    wav_bytes = await audio.read()

    try:
        result = run_diagnosis(
            json_data=metadata,
            wav_bytes=wav_bytes,
            wav_filename=audio.filename
        )
        return JSONResponse(content=result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Error en diagnóstico")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS Y EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/metrics", tags=["Evaluación"])
async def evaluate_model():
    try:
        result = run_evaluation()
        clean_result = _sanitize_json_payload(result)
        return JSONResponse(content=clean_result)
    except Exception as e:
        logger.exception("Error en evaluación")
        raise HTTPException(status_code=500, detail=f"Error evaluando métricas: {str(e)}")

@app.get("/api/v1/metrics/history", tags=["Evaluación"])
async def metrics_history():
    """Retorna el historial completo de evaluaciones del modelo."""
    return JSONResponse(content=_sanitize_json_payload(get_metrics_history()))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
