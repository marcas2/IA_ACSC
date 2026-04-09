"""
services/train_service.py
SERVICIO 1 — Entrenamiento inicial con los audios del servidor local.

Flujo:
  1. Lista todos los JSON disponibles en el servidor local
  2. Descarga JSON + audio principal + ECG (si existe)
  3. Extrae características de cada muestra
  4. Entrena el modelo (batch completo)
  5. Evalúa con validación cruzada y reporta métricas
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

from config import LABEL_ANOMALIA, LABEL_NORMAL
from models.valvulopatia_model import ValvulopatiaModel, get_model, reload_model
from utils.feature_extractor import build_full_feature_vector, load_audio_from_bytes
from utils.local_server import (
    download_audio_json, download_audio_principal,
    download_audio_ecg, list_all_available_samples
)

logger = logging.getLogger(__name__)


def _extract_label_from_metadata(metadata: dict) -> str:
    """Determina la etiqueta de una muestra desde el JSON."""
    estado = metadata.get("diagnostico", {}).get("estado", "").strip()
    categoria = metadata.get("diagnostico", {}).get("categoria_anomalia")

    if estado.lower() == "normal" and categoria is None:
        return LABEL_NORMAL
    else:
        return LABEL_ANOMALIA


def load_dataset_from_server() -> Tuple[np.ndarray, List[str], List[dict]]:
    """
    Descarga y procesa todas las muestras del servidor local.
    
    Retorna:
      X:        (n_samples, n_features)
      labels:   lista de "Normal" / "Anomalia"
      metadata_list: lista de los JSONs originales
    """
    samples = list_all_available_samples()
    if not samples:
        raise ValueError(
            "No se encontraron muestras en el servidor local. "
            "Verifique la conexión con http://172.16.10.200:5002"
        )

    X_list: List[np.ndarray] = []
    labels:  List[str]       = []
    metas:   List[dict]      = []
    skipped: int             = 0

    logger.info(f"Procesando {len(samples)} muestras del servidor local…")

    for i, sample in enumerate(samples):
        try:
            # 1. JSON clínico
            metadata = download_audio_json(sample["json_file"])
            if metadata is None:
                logger.warning(f"[{i+1}] JSON no disponible: {sample['json_file']}")
                skipped += 1
                continue

            # 2. Audio principal
            wav_bytes = download_audio_principal(sample["wav_file"])
            if wav_bytes is None:
                logger.warning(f"[{i+1}] Audio no disponible: {sample['wav_file']}")
                skipped += 1
                continue

            y_principal = load_audio_from_bytes(wav_bytes)
            if y_principal is None:
                skipped += 1
                continue

            # 3. ECG (opcional)
            y_ecg = None
            if sample.get("ecg_file"):
                ecg_bytes = download_audio_ecg(sample["ecg_file"])
                if ecg_bytes:
                    y_ecg = load_audio_from_bytes(ecg_bytes)

            # 4. Vector de características
            feat_vector = build_full_feature_vector(y_principal, metadata, y_ecg)
            label       = _extract_label_from_metadata(metadata)

            X_list.append(feat_vector)
            labels.append(label)
            metas.append(metadata)

            logger.info(
                f"[{i+1}/{len(samples)}] {sample['wav_file']} → {label} "
                f"({feat_vector.shape[0]} features)"
            )

        except Exception as e:
            logger.error(f"Error procesando {sample}: {e}")
            skipped += 1

    if not X_list:
        raise ValueError("Ninguna muestra pudo procesarse correctamente.")

    X = np.vstack(X_list)
    logger.info(
        f"Dataset listo: {X.shape[0]} muestras, {X.shape[1]} features. "
        f"Omitidas: {skipped}. "
        f"Normales: {labels.count(LABEL_NORMAL)} | "
        f"Anomalías: {labels.count(LABEL_ANOMALIA)}"
    )
    return X, labels, metas


def evaluate_with_cross_validation(X: np.ndarray, labels: List[str]) -> Dict:
    """
    Evalúa el modelo con validación cruzada estratificada (k=5 o menos si hay pocas muestras).
    Retorna métricas detalladas.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    n_splits = min(5, len(np.unique(y, return_counts=True)[1].min(), 5))
    # Asegurar al menos 2 splits
    n_splits = max(2, n_splits)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        ))
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["accuracy", "f1_weighted", "roc_auc"],
        return_train_score=True
    )

    return {
        "cv_folds": n_splits,
        "accuracy_mean":  round(float(np.mean(scores["test_accuracy"])), 4),
        "accuracy_std":   round(float(np.std(scores["test_accuracy"])),  4),
        "f1_mean":        round(float(np.mean(scores["test_f1_weighted"])), 4),
        "roc_auc_mean":   round(float(np.mean(scores["test_roc_auc"])),  4),
        "train_acc_mean": round(float(np.mean(scores["train_accuracy"])), 4),
    }


def run_full_training() -> Dict:
    """
    Punto de entrada del SERVICIO 1.
    Descarga datos, entrena el modelo y retorna métricas.
    """
    logger.info("═══ INICIO ENTRENAMIENTO COMPLETO ═══")

    # Cargar dataset
    X, labels, metas = load_dataset_from_server()

    normal_count = labels.count(LABEL_NORMAL)
    anomalia_count = labels.count(LABEL_ANOMALIA)
    if normal_count == 0 or anomalia_count == 0:
        raise ValueError(
            "No se puede entrenar el modelo porque solo hay una clase en el dataset. "
            f"Normal={normal_count}, Anomalia={anomalia_count}. "
            "Agregue muestras de la clase faltante y vuelva a ejecutar /api/v1/train/full."
        )

    # Validación cruzada ANTES de entrenar el modelo final
    cv_metrics = {}
    if len(X) >= 4:
        try:
            cv_metrics = evaluate_with_cross_validation(X, labels)
            logger.info(f"CV Accuracy: {cv_metrics['accuracy_mean']:.3f} ± {cv_metrics['accuracy_std']:.3f}")
        except Exception as e:
            logger.warning(f"CV no completada: {e}")

    # Entrenamiento del modelo final con todos los datos
    model = get_model()
    train_metrics = model.fit(X, labels)
    reload_model()

    result = {
        "status": "success",
        "message": "Modelo entrenado y guardado exitosamente",
        "dataset": {
            "total_muestras": len(X),
            "normales":   labels.count(LABEL_NORMAL),
            "anomalias":  labels.count(LABEL_ANOMALIA),
            "n_features": int(X.shape[1])
        },
        "train_metrics": train_metrics,
        "cross_validation": cv_metrics
    }

    logger.info(f"═══ ENTRENAMIENTO COMPLETADO: {result} ═══")
    return result
