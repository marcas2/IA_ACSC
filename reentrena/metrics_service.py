import json
import logging
import math
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold

from config import DATA_DIR, METRICS_FILE
from models.valvulopatia_model import get_model
from services.train_service import load_dataset_from_server

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def _safe_float(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _safe_round(value, digits=4):
    v = _safe_float(value)
    return round(v, digits) if v is not None else None


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]

    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]

    if isinstance(value, np.ndarray):
        return [_sanitize_for_json(v) for v in value.tolist()]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        fval = float(value)
        return fval if math.isfinite(fval) else None

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def _save_metrics(metrics: Dict):
    history = []
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE) as f:
                history = _sanitize_for_json(json.load(f))
        except Exception:
            logger.warning("No se pudo leer metrics_history.json, se recreará.")
            history = []

    history.append(_sanitize_for_json(metrics))

    with open(METRICS_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False, allow_nan=False)


def generate_confusion_matrix_plot(y_true: List[int], y_pred: List[int], classes: List[str]) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión - ValvIA")
    plt.tight_layout()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"confusion_matrix_{ts}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def generate_roc_curve_plot(y_true: List[int], y_scores: List[float], roc_auc: float) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curva ROC - ValvIA")
    ax.legend(loc="lower right")
    plt.tight_layout()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"roc_curve_{ts}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def run_evaluation(use_server_data: bool = True) -> Dict:
    """
    Evalúa el modelo contra los datos del servidor local usando
    validación cruzada estratificada.
    """
    model = get_model()
    if not model.is_ready:
        return {
            "status": "error",
            "message": "El modelo no está entrenado. Ejecute /api/v1/train/full primero."
        }

    logger.info("Cargando dataset para evaluación…")
    X, labels, _ = load_dataset_from_server()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)

    pred_labels, pred_probs = model.predict(X)
    y_pred = le.transform(pred_labels)

    acc = accuracy_score(y, y_pred)

    roc_auc = None
    if len(np.unique(y)) == 2:
        try:
            roc_auc = float(roc_auc_score(y, pred_probs))
            if not math.isfinite(roc_auc):
                roc_auc = None
        except Exception:
            roc_auc = None

    report_dict = classification_report(
        y,
        y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y, y_pred).tolist()

    cm_path = generate_confusion_matrix_plot(
        y.tolist(), y_pred.tolist(), list(le.classes_)
    )

    roc_path = None
    if roc_auc is not None:
        try:
            roc_path = generate_roc_curve_plot(y.tolist(), pred_probs, roc_auc)
        except Exception:
            roc_path = None

    cv_metrics = {}

    class_counts = np.bincount(y)
    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0

    if len(X) >= 4 and len(np.unique(y)) == 2 and min_class_count >= 2:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import cross_validate

            n_splits = min(5, min_class_count)

            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))
            ])

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            scores = cross_validate(
                pipe,
                X,
                y,
                cv=cv,
                scoring=["accuracy", "f1_weighted", "roc_auc"],
                return_train_score=False,
                error_score=np.nan
            )

            def safe_metric(arr):
                val = float(np.nanmean(arr))
                return round(val, 4) if math.isfinite(val) else None

            def safe_std(arr):
                val = float(np.nanstd(arr))
                return round(val, 4) if math.isfinite(val) else None

            cv_metrics = {
                "enabled": True,
                "folds": n_splits,
                "cv_accuracy_mean": safe_metric(scores["test_accuracy"]),
                "cv_accuracy_std": safe_std(scores["test_accuracy"]),
                "cv_f1_mean": safe_metric(scores["test_f1_weighted"]),
                "cv_roc_auc_mean": safe_metric(scores["test_roc_auc"]),
            }
        except Exception as e:
            cv_metrics = {
                "enabled": False,
                "reason": f"Validación cruzada omitida por error: {str(e)}"
            }
    else:
        cv_metrics = {
            "enabled": False,
            "reason": (
                "Validación cruzada omitida: se requieren al menos 2 muestras por clase. "
                f"Distribución actual: {dict(zip(le.classes_.tolist(), class_counts.tolist()))}"
            )
        }

    def safe_round(v):
        try:
            v = float(v)
            return round(v, 4) if math.isfinite(v) else None
        except Exception:
            return None

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "success",
        "dataset": {
            "total": int(len(X)),
            "distribucion": dict(zip(*np.unique(labels, return_counts=True))),
            "n_features": int(X.shape[1])
        },
        "metricas_generales": {
            "accuracy": safe_round(acc),
            "roc_auc": safe_round(roc_auc),
        },
        "reporte_clasificacion": {
            k: {
                "precision": safe_round(v.get("precision")),
                "recall": safe_round(v.get("recall")),
                "f1_score": safe_round(v.get("f1-score")),
                "support": int(v.get("support", 0))
            }
            for k, v in report_dict.items()
            if isinstance(v, dict)
        },
        "matriz_confusion": cm,
        "clases_orden": list(le.classes_),
        "validacion_cruzada": cv_metrics,
        "archivos_generados": {
            "confusion_matrix": cm_path,
            "roc_curve": roc_path
        }
    }

    metrics = _sanitize_for_json(metrics)
    _save_metrics(metrics)
    logger.info(f"Evaluación completada. Accuracy: {acc:.3f}")
    return metrics


def get_metrics_history() -> List[Dict]:
    """Retorna el historial completo de evaluaciones."""
    if not os.path.exists(METRICS_FILE):
        return []

    try:
        with open(METRICS_FILE) as f:
            history = json.load(f)
        return _sanitize_for_json(history)
    except Exception:
        logger.warning("No se pudo leer metrics_history.json; se retornará lista vacía.")
        return []