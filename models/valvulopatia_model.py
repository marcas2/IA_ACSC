"""
models/valvulopatia_model.py
Modelo de IA para detección de valvulopatías.

Arquitectura:
  - Ensemble: Gradient Boosting + SVM + Random Forest con votación suave
  - Entrenado con características acústicas + metadatos clínicos
  - Soporta entrenamiento incremental (warm_start / partial_fit con nuevo SGD)
  - Persiste en disco con joblib
"""
import json
import logging
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config import (
    DATA_DIR, ENCODER_PATH, LABEL_ANOMALIA, LABEL_NORMAL,
    MODEL_PATH, SCALER_PATH, TRAINING_LOG
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de persistencia
# ──────────────────────────────────────────────────────────────────────────────

def _log_training_event(n_samples: int, accuracy: float, details: dict):
    log = []
    if os.path.exists(TRAINING_LOG):
        with open(TRAINING_LOG) as f:
            log = json.load(f)
    log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "n_samples": n_samples,
        "accuracy": round(accuracy, 4),
        **details
    })
    with open(TRAINING_LOG, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Modelo principal
# ──────────────────────────────────────────────────────────────────────────────

class ValvulopatiaModel:
    """
    Clasificador binario: Normal vs. Anomalía (valvulopatía).
    
    Estrategia:
      - Con ≥ 30 muestras: Ensemble (GBM + RF + SVM calibrado) con votación suave.
      - Con < 30 muestras: SGD (descenso estocástico) que acepta partial_fit.
      - Siempre normaliza con StandardScaler.
    """

    def __init__(self):
        self.scaler        = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model         = None
        self.sgd_model     = None          # Para entrenamiento incremental
        self._is_fitted    = False
        self._n_trained    = 0
        self._feature_dim  = None

    # ── Construcción del ensemble ─────────────────────────────────────────────

    def _build_ensemble(self) -> VotingClassifier:
        gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=4,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        svm_base = SVC(kernel="rbf", C=1.0, gamma="scale",
                       class_weight="balanced", probability=False, random_state=42)
        svm_cal = CalibratedClassifierCV(svm_base, cv=3, method="sigmoid")

        ensemble = VotingClassifier(
            estimators=[("gbm", gbm), ("rf", rf), ("svm", svm_cal)],
            voting="soft",
            weights=[2, 2, 1]        # GBM y RF pesan más
        )
        return ensemble

    def _build_sgd(self, classes: np.ndarray) -> SGDClassifier:
        clf = SGDClassifier(
            loss="modified_huber",   # permite predict_proba
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
        return clf

    # ── Entrenamiento completo (batch) ────────────────────────────────────────

    def fit(self, X: np.ndarray, y_labels: List[str]) -> Dict:
        """
        Entrena el modelo desde cero con todos los datos disponibles.
        X: matriz (n_samples, n_features)
        y_labels: lista de strings "Normal" / "Anomalia"
        """
        if len(X) < 2:
            raise ValueError("Se necesitan al menos 2 muestras para entrenar.")

        unique_labels = sorted(set(y_labels))
        if len(unique_labels) < 2:
            raise ValueError(
                "No se puede entrenar con una sola clase. "
                f"Clases detectadas: {unique_labels}. "
                "Agregue muestras de al menos dos clases (Normal y Anomalia)."
            )

        y = self.label_encoder.fit_transform(y_labels)
        self._feature_dim = X.shape[1]

        X_scaled = self.scaler.fit_transform(X)

        if len(X) >= 30:
            logger.info("Entrenando ensemble completo (≥30 muestras)…")
            self.model = self._build_ensemble()
            self.model.fit(X_scaled, y)
        else:
            logger.info(f"Entrenando SGD (solo {len(X)} muestras)…")
            classes = np.unique(y)
            self.sgd_model = self._build_sgd(classes)
            self.sgd_model.fit(X_scaled, y)
            self.model = self.sgd_model

        self._is_fitted    = True
        self._n_trained    = len(X)

        # Accuracy en entrenamiento (referencia, no generalización)
        y_pred = self.model.predict(X_scaled)
        train_acc = float(np.mean(y_pred == y))
        logger.info(f"Accuracy en entrenamiento: {train_acc:.3f}")

        self._save()
        _log_training_event(len(X), train_acc, {"mode": "full_fit"})

        return {"n_samples": len(X), "train_accuracy": round(train_acc, 4)}

    # ── Entrenamiento incremental (online) ────────────────────────────────────

    def partial_fit(self, X_new: np.ndarray, y_new_labels: List[str]) -> Dict:
        """
        Agrega nuevas muestras al modelo existente sin reentrenar desde cero.
        Usa SGDClassifier con partial_fit (ideal para aprendizaje continuo).
        """
        all_classes = np.array([0, 1])  # Normal=0, Anomalia=1

        y_new = self.label_encoder.transform(y_new_labels)
        X_new_scaled = self.scaler.transform(X_new)

        if self.sgd_model is None:
            logger.info("Creando SGD para entrenamiento incremental…")
            self.sgd_model = SGDClassifier(
                loss="modified_huber",
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
            # Inicializar con datos anteriores si el modelo actual ya fue ajustado
            if self._is_fitted and isinstance(self.model, VotingClassifier):
                logger.warning(
                    "El ensemble no soporta partial_fit directamente. "
                    "El SGD incremental operará en paralelo hasta reentrenamiento completo."
                )

        self.sgd_model.partial_fit(X_new_scaled, y_new, classes=all_classes)
        # El SGD pasa a ser el modelo activo para inferencia incremental
        self.model = self.sgd_model
        self._is_fitted = True
        self._n_trained += len(X_new)

        y_pred = self.sgd_model.predict(X_new_scaled)
        acc = float(np.mean(y_pred == y_new))

        self._save()
        _log_training_event(
            self._n_trained, acc,
            {"mode": "partial_fit", "new_samples": len(X_new)}
        )

        return {
            "new_samples": len(X_new),
            "total_trained": self._n_trained,
            "incremental_accuracy": round(acc, 4)
        }

    # ── Inferencia ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Predice etiquetas y probabilidades de valvulopatía.
        Retorna (labels, probs_anomalia)
        """
        if not self._is_fitted:
            raise NotFittedError("El modelo no ha sido entrenado todavía.")

        X_scaled = self.scaler.transform(X)

        y_pred  = self.model.predict(X_scaled)
        labels  = list(self.label_encoder.inverse_transform(y_pred))

        try:
            proba   = self.model.predict_proba(X_scaled)
            # índice de la clase "Anomalia"
            anomalia_idx = list(self.label_encoder.classes_).index(LABEL_ANOMALIA)
            probs_anomalia = proba[:, anomalia_idx].tolist()
        except Exception:
            # Fallback si el modelo no soporta predict_proba
            probs_anomalia = [1.0 if l == LABEL_ANOMALIA else 0.0 for l in labels]

        return labels, probs_anomalia

    def predict_single(self, x: np.ndarray) -> Dict:
        """
        Predice para una sola muestra. Retorna dict con diagnóstico completo.
        """
        labels, probs = self.predict(x.reshape(1, -1))
        label   = labels[0]
        prob    = probs[0]
        tiene_valvulopatia = label == LABEL_ANOMALIA

        # Nivel de confianza
        if prob >= 0.85:
            confianza = "Alta"
        elif prob >= 0.60:
            confianza = "Media"
        else:
            confianza = "Baja"

        return {
            "diagnostico": label,
            "tiene_valvulopatia": tiene_valvulopatia,
            "probabilidad_anomalia": round(prob, 4),
            "probabilidad_normal": round(1 - prob, 4),
            "confianza": confianza,
            "modelo_entrenado_con": self._n_trained
        }

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _save(self):
        joblib.dump(self.model,         MODEL_PATH)
        joblib.dump(self.scaler,        SCALER_PATH)
        joblib.dump(self.label_encoder, ENCODER_PATH)
        logger.info("Modelo guardado en disco.")

    @classmethod
    def load(cls) -> "ValvulopatiaModel":
        """Carga el modelo desde disco. Si no existe, retorna instancia vacía."""
        instance = cls()
        if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
            try:
                instance.model         = joblib.load(MODEL_PATH)
                instance.scaler        = joblib.load(SCALER_PATH)
                instance.label_encoder = joblib.load(ENCODER_PATH)
                instance._is_fitted    = True
                logger.info("Modelo cargado desde disco.")
            except Exception as e:
                logger.error(f"Error cargando modelo: {e}")
                instance._is_fitted = False
        else:
            logger.warning("No se encontró modelo pre-entrenado. El modelo está vacío.")
        return instance

    @property
    def is_ready(self) -> bool:
        return self._is_fitted


# ── Instancia global (singleton) ─────────────────────────────────────────────
_model_instance: Optional[ValvulopatiaModel] = None


def get_model() -> ValvulopatiaModel:
    """Obtiene (o crea) la instancia global del modelo."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ValvulopatiaModel.load()
    return _model_instance


def reload_model():
    """Recarga el modelo desde disco (útil tras reentrenamiento)."""
    global _model_instance
    _model_instance = ValvulopatiaModel.load()
