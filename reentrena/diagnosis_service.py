"""
services/diagnosis_service.py
SERVICIO 3 — Recibe 1 audio WAV + JSON, genera diagnóstico de valvulopatía
y almacena el audio en el servidor local (SIN etiqueta de diagnóstico conocida).
"""
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from models.valvulopatia_model import get_model
from utils.feature_extractor import build_full_feature_vector, load_audio_from_bytes
from utils.local_server import upload_file
from config import REMOTE_FOLDERS

logger = logging.getLogger(__name__)


def run_diagnosis(
    json_data: dict,
    wav_bytes: bytes,
    wav_filename: str,
) -> Dict:
    """
    Punto de entrada del SERVICIO 3.
    
    1. Valida que el modelo esté disponible
    2. Almacena el audio en el servidor local (carpeta Audios)
    3. Almacena el JSON en el servidor local (carpeta audios-json)
    4. Extrae características del audio
    5. Realiza la inferencia
    6. Retorna el diagnóstico detallado
    
    El JSON de entrada NO necesita tener 'diagnostico.estado' 
    (el modelo lo determinará).
    """
    model = get_model()
    if not model.is_ready:
        raise RuntimeError(
            "El modelo no está entrenado. "
            "Ejecute primero el endpoint /api/v1/train/full"
        )

    # ── 1. Almacenamiento en servidor local ───────────────────────────────────
    # Audio principal → carpeta Audios/
    audio_uploaded = upload_file(
        REMOTE_FOLDERS["audio_principal"],
        wav_filename,
        wav_bytes,
        content_type="audio/wav"
    )

    # JSON con metadatos del paciente → carpeta audios-json/
    json_filename = wav_filename.replace(".wav", ".json")
    json_uploaded = upload_file(
        REMOTE_FOLDERS["audio_json"],
        json_filename,
        json.dumps(json_data, ensure_ascii=False).encode("utf-8"),
        content_type="application/json"
    )

    # ── 2. Extracción de características ──────────────────────────────────────
    y_audio = load_audio_from_bytes(wav_bytes)
    if y_audio is None:
        raise ValueError("No se pudo procesar el archivo de audio.")

    feat_vector = build_full_feature_vector(y_audio, json_data, y_ecg=None)

    # ── 3. Inferencia ─────────────────────────────────────────────────────────
    resultado = model.predict_single(feat_vector)

    # ── 4. Enriquecer respuesta con contexto clínico ──────────────────────────
    paciente  = json_data.get("paciente", {})
    meta      = json_data.get("metadata", {})
    diag_meta = json_data.get("diagnostico", {})

    response = {
        "timestamp_diagnostico": datetime.utcnow().isoformat(),
        "archivo_analizado": wav_filename,

        # Datos del paciente
        "paciente": {
            "edad":     meta.get("edad"),
            "genero":   paciente.get("genero"),
            "peso_kg":  paciente.get("peso_kg"),
            "altura_cm": paciente.get("altura_cm"),
        },

        # Contexto clínico
        "foco_auscultacion": diag_meta.get("foco_auscultacion"),

        # Resultado de la IA
        "resultado_ia": resultado,

        # Recomendación clínica basada en probabilidad
        "recomendacion": _generate_recommendation(resultado),

        # Estado del almacenamiento
        "almacenamiento": {
            "audio_guardado": audio_uploaded,
            "json_guardado":  json_uploaded,
        }
    }

    logger.info(
        f"Diagnóstico generado para {wav_filename}: "
        f"{resultado['diagnostico']} "
        f"(prob={resultado['probabilidad_anomalia']:.3f})"
    )

    return response


def _generate_recommendation(resultado: Dict) -> str:
    """Genera una recomendación clínica basada en el resultado de la IA."""
    prob  = resultado.get("probabilidad_anomalia", 0)
    conf  = resultado.get("confianza", "Baja")
    tiene = resultado.get("tiene_valvulopatia", False)

    if tiene:
        if conf == "Alta":
            return (
                "⚠️ Alta probabilidad de valvulopatía detectada. "
                "Se recomienda evaluación cardiológica urgente con "
                "ecocardiograma Doppler."
            )
        elif conf == "Media":
            return (
                "⚠️ Señales compatibles con valvulopatía. "
                "Se recomienda seguimiento cardiológico y estudio complementario."
            )
        else:
            return (
                "⚠️ Posibles alteraciones cardiacas. "
                "Se sugiere monitoreo y consulta con especialista."
            )
    else:
        if conf == "Alta":
            return (
                "✅ Sonido cardiaco dentro de parámetros normales con alta confianza. "
                "Control rutinario recomendado."
            )
        else:
            return (
                "✅ Sin evidencia de valvulopatía. "
                "Se recomienda seguimiento clínico de rutina."
            )
