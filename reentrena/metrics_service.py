"""
services/metrics_service.py
Evaluación del modelo: accuracy, F1, ROC-AUC, matriz de confusión.
Puede ejecutarse en cualquier momento para saber qué tan bien acierta la IA.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedKFold

from config import (
    DATA_DIR, LABEL_ANOMALIA, LABEL_NORMAL, METRICS_FILE
)
from models.valvulopatia_model import get_model
from services.train_service import load_dataset_from_server

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def _save_metrics(metrics: Dict):
    """Persiste el historial de métricas en JSON."""
    history = []
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            history = json.load(f)
    history.append(metrics)
    with open(METRICS_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def generate_confusion_matrix_plot(
    y_true: List[int], y_pred: List[int],
    classes: List[str]
) -> str:
    """Genera y guarda imagen de la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión - ValvIA")
    plt.tight_layout()

    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"confusion_matrix_{ts}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def generate_roc_curve_plot(
    y_true: List[int], y_scores: List[float], roc_auc: float
) -> str:
    """Genera y guarda imagen de la curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curva ROC - ValvIA")
    ax.legend(loc="lower right")
    plt.tight_layout()

    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"roc_curve_{ts}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def run_evaluation(use_server_data: bool = True) -> Dict:
    """
    Evalúa el modelo contra los datos del servidor local usando
    validación cruzada estratificada.
    
    Retorna métricas completas:
      - accuracy, precision, recall, F1 (por clase y global)
      - ROC-AUC
      - Matriz de confusión
      - Historial de rendimiento
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
    y  = le.fit_transform(labels)

    # ── Predicción con el modelo entrenado ────────────────────────────────────
    pred_labels, pred_probs = model.predict(X)
    y_pred = le.transform(pred_labels)

    acc = accuracy_score(y, y_pred)

    # ROC-AUC (solo si hay 2 clases)
    roc_auc = None
    if len(np.unique(y)) == 2:
        try:
            roc_auc = float(roc_auc_score(y, pred_probs))
        except Exception:
            pass

    # Reporte de clasificación
    report_dict = classification_report(
        y, y_pred,
        target_names=le.classes_,
        output_dict=True
    )

    # Matriz de confusión
    cm = confusion_matrix(y, y_pred).tolist()

    # Plots
    cm_path  = generate_confusion_matrix_plot(y.tolist(), y_pred.tolist(), list(le.classes_))
    roc_path = None
    if roc_auc is not None:
        roc_path = generate_roc_curve_plot(y.tolist(), pred_probs, roc_auc)

    # ── Validación cruzada ────────────────────────────────────────────────────
    cv_metrics = {}
    if len(X) >= 4:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_validate

        n_splits = min(5, min(np.bincount(y)))
        n_splits = max(2, n_splits)

        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring=["accuracy", "f1_weighted", "roc_auc"],
            return_train_score=False
        )
        cv_metrics = {
            "folds": n_splits,
            "cv_accuracy_mean":  round(float(np.mean(scores["test_accuracy"])), 4),
            "cv_accuracy_std":   round(float(np.std(scores["test_accuracy"])),  4),
            "cv_f1_mean":        round(float(np.mean(scores["test_f1_weighted"])), 4),
            "cv_roc_auc_mean":   round(float(np.mean(scores["test_roc_auc"])),  4),
        }

    # ── Distribución del dataset ──────────────────────────────────────────────
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "success",
        "dataset": {
            "total": len(X),
            "distribucion": dist,
            "n_features": int(X.shape[1])
        },
        "metricas_generales": {
            "accuracy":       round(float(acc), 4),
            "roc_auc":        round(roc_auc, 4) if roc_auc else None,
        },
        "reporte_clasificacion": {
            k: {
                "precision": round(v["precision"], 4),
                "recall":    round(v["recall"], 4),
                "f1_score":  round(v["f1-score"], 4),
                "support":   int(v["support"])
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

    _save_metrics(metrics)
    logger.info(f"Evaluación completada. Accuracy: {acc:.3f}")
    return metrics


def get_metrics_history() -> List[Dict]:
    """Retorna el historial completo de evaluaciones."""
    if not os.path.exists(METRICS_FILE):
        return []
    with open(METRICS_FILE) as f:
        return json.load(f)
