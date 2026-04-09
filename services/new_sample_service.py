"""
services/new_sample_service.py
SERVICIO 2 — Recibe nueva muestra (4 audios + JSON), la almacena en el
servidor local y actualiza el modelo con entrenamiento incremental.
"""
import json
import logging
from typing import Dict, Optional

import numpy as np

from config import LABEL_ANOMALIA, LABEL_NORMAL
from models.valvulopatia_model import get_model, reload_model
from utils.feature_extractor import build_full_feature_vector, load_audio_from_bytes
from utils.local_server import upload_training_sample

logger = logging.getLogger(__name__)


def _extract_label_from_metadata(metadata: dict) -> str:
    estado    = metadata.get("diagnostico", {}).get("estado", "").strip()
    categoria = metadata.get("diagnostico", {}).get("categoria_anomalia")
    return LABEL_NORMAL if (estado.lower() == "normal" and categoria is None) else LABEL_ANOMALIA


def process_new_training_sample(
    json_data: dict,
    wav_principal_bytes: bytes,
    wav_principal_name:  str,
    wav_ecg_bytes:  Optional[bytes] = None,
    wav_ecg_name:   Optional[str]   = None,
    wav_ecg1_bytes: Optional[bytes] = None,
    wav_ecg1_name:  Optional[str]   = None,
    wav_ecg2_bytes: Optional[bytes] = None,
    wav_ecg2_name:  Optional[str]   = None,
) -> Dict:
    """
    Punto de entrada del SERVICIO 2.
    
    1. Valida que el JSON tenga diagnóstico (estado Normal/Anomalia)
    2. Sube los archivos al servidor local en sus carpetas correspondientes
    3. Extrae características del audio
    4. Actualiza el modelo con partial_fit (aprendizaje incremental)
    5. Retorna resumen de la operación
    """
    # ── 1. Validación del JSON ────────────────────────────────────────────────
    diagnostico = json_data.get("diagnostico", {})
    estado = diagnostico.get("estado", "").strip()
    if not estado:
        raise ValueError(
            "El JSON debe incluir 'diagnostico.estado' "
            "(ej: 'Normal' o nombre de anomalía)"
        )

    label = _extract_label_from_metadata(json_data)
    logger.info(f"Nueva muestra recibida: {wav_principal_name} → {label}")

    # ── 2. Subida al servidor local ───────────────────────────────────────────
    # Construir nombre del JSON desde el nombre del wav
    json_filename = wav_principal_name.replace(".wav", ".json")

    upload_results = upload_training_sample(
        json_filename=json_filename,
        json_data=json.dumps(json_data, ensure_ascii=False).encode("utf-8"),
        wav_principal=wav_principal_bytes,
        wav_principal_name=wav_principal_name,
        wav_ecg=wav_ecg_bytes,   wav_ecg_name=wav_ecg_name,
        wav_ecg1=wav_ecg1_bytes, wav_ecg1_name=wav_ecg1_name,
        wav_ecg2=wav_ecg2_bytes, wav_ecg2_name=wav_ecg2_name,
    )

    all_uploaded = all(upload_results.values())
    if not all_uploaded:
        failed = [k for k, v in upload_results.items() if not v]
        logger.warning(f"Archivos no subidos: {failed}")

    # ── 3. Extracción de características ─────────────────────────────────────
    y_principal = load_audio_from_bytes(wav_principal_bytes)
    if y_principal is None:
        raise ValueError("No se pudo procesar el archivo de audio principal.")

    y_ecg = None
    if wav_ecg_bytes:
        y_ecg = load_audio_from_bytes(wav_ecg_bytes)

    feat_vector = build_full_feature_vector(y_principal, json_data, y_ecg)
    X_new = feat_vector.reshape(1, -1)

    # ── 4. Entrenamiento incremental ──────────────────────────────────────────
    model = get_model()

    incremental_result = {}
    if model.is_ready:
        try:
            incremental_result = model.partial_fit(X_new, [label])
            reload_model()
            logger.info(f"Modelo actualizado incrementalmente. Muestras totales: {incremental_result.get('total_trained')}")
        except Exception as e:
            logger.error(f"Error en partial_fit: {e}. La muestra se guardó pero el modelo no se actualizó.")
            incremental_result = {"error": str(e)}
    else:
        logger.warning(
            "El modelo aún no está entrenado. "
            "La muestra fue almacenada. Ejecute el entrenamiento completo primero."
        )
        incremental_result = {"warning": "Modelo no inicializado. Muestra almacenada para entrenamiento futuro."}

    return {
        "status": "success",
        "archivo": wav_principal_name,
        "label_asignado": label,
        "upload_results": upload_results,
        "all_files_uploaded": all_uploaded,
        "features_extraidas": int(feat_vector.shape[0]),
        "actualizacion_modelo": incremental_result
    }
