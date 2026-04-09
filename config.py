"""
Configuración central del sistema de detección de valvulopatías.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Servidor local de almacenamiento de audio ────────────────────────────────
LOCAL_SERVER_BASE_URL = os.getenv("LOCAL_SERVER_BASE_URL", "http://172.16.10.200:5002")

# Rutas de carpetas en el servidor local (Apache directory listing)
REMOTE_FOLDERS = {
    "audio_principal": "Audios",
    "audio_json":      "audios-json",
    "audio_ecg":       "ECG",
    "audio_ecg_1":     "ECG_1",
    "audio_ecg_2":     "ECG_2",
}

# ─── Parámetros de extracción de características ──────────────────────────────
SAMPLE_RATE        = 4000          # Hz recomendado para fonocardiograma
N_MFCC             = 40            # Coeficientes MFCC
N_MELS             = 64            # Bandas Mel
HOP_LENGTH         = 512
N_FFT              = 2048
AUDIO_DURATION_SEC = 10            # Duración máxima de audio a procesar (segundos)

# ─── Rutas locales (en la nube) ───────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR       = os.path.join(BASE_DIR, "models")
DATA_DIR        = os.path.join(BASE_DIR, "data")
METRICS_FILE    = os.path.join(DATA_DIR, "metrics_history.json")
TRAINING_LOG    = os.path.join(DATA_DIR, "training_log.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ─── Nombres de modelos persistidos ──────────────────────────────────────────
MODEL_PATH      = os.path.join(MODEL_DIR, "valvulopatia_model.joblib")
SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH    = os.path.join(MODEL_DIR, "label_encoder.joblib")

# ─── Clases de diagnóstico ────────────────────────────────────────────────────
LABEL_NORMAL   = "Normal"
LABEL_ANOMALIA = "Anomalia"

# ─── API ──────────────────────────────────────────────────────────────────────
API_TITLE   = "ValvIA - Sistema de Detección de Valvulopatías"
API_VERSION = "1.0.0"
API_HOST    = os.getenv("API_HOST", "0.0.0.0")
API_PORT    = int(os.getenv("API_PORT", "8000"))
