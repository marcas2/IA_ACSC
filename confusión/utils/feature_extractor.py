"""
utils/feature_extractor.py
Extracción de características acústicas de sonidos cardiacos.
Combina MFCC, Mel-Spectrograma, ZCR, RMS y metadatos del JSON clínico.
"""
import io
import numpy as np
import librosa
from typing import Optional
import logging

from config import (
    SAMPLE_RATE, N_MFCC, N_MELS,
    HOP_LENGTH, N_FFT, AUDIO_DURATION_SEC
)

logger = logging.getLogger(__name__)


def load_audio_from_bytes(audio_bytes: bytes) -> Optional[np.ndarray]:
    """Carga un audio WAV desde bytes y lo normaliza a SAMPLE_RATE."""
    try:
        y, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=SAMPLE_RATE,
            mono=True,
            duration=AUDIO_DURATION_SEC
        )
        # Padding si el audio es muy corto
        target_len = SAMPLE_RATE * AUDIO_DURATION_SEC
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        return y
    except Exception as e:
        logger.error(f"Error cargando audio: {e}")
        return None


def extract_audio_features(y: np.ndarray) -> np.ndarray:
    """
    Extrae un vector de características del audio.
    
    Retorna un vector 1-D de dimensión fija con:
      - MFCC (mean + std)  → 2×N_MFCC
      - Delta MFCC (mean)  → N_MFCC
      - Mel-spectrogram (mean por banda) → N_MELS
      - Chroma (mean)      → 12
      - Spectral Centroid (mean, std) → 2
      - Spectral Bandwidth (mean)     → 1
      - Spectral Rolloff (mean)       → 1
      - ZCR (mean)         → 1
      - RMS Energy (mean, std) → 2
    Total: ~238 características
    """
    feats = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))

    # Delta MFCC (primera derivada)
    delta_mfcc = librosa.feature.delta(mfcc)
    feats.extend(np.mean(delta_mfcc, axis=1))

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats.extend(np.mean(mel_db, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.extend(np.mean(chroma, axis=1))

    # Spectral features
    sc = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE,
                                            n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.append(float(np.mean(sc)))
    feats.append(float(np.std(sc)))

    sb = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.append(float(np.mean(sb)))

    sr_feat = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE,
                                                n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.append(float(np.mean(sr_feat)))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    feats.append(float(np.mean(zcr)))

    # RMS Energy
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    feats.append(float(np.mean(rms)))
    feats.append(float(np.std(rms)))

    return np.array(feats, dtype=np.float32)


def extract_metadata_features(metadata: dict) -> np.ndarray:
    """
    Convierte el JSON clínico en un vector numérico.
    
    Campos usados:
      - edad
      - genero (M=0, F=1, O=2)
      - peso_kg
      - altura_cm
      - imc (calculado)
      - foco_auscultacion (codificado)
      - enfermedades_base (si está presente, binario)
    """
    feats = []

    # Datos del paciente
    paciente = metadata.get("paciente", {})
    edad     = float(metadata.get("metadata", {}).get("edad", 0) or 0)
    genero_raw = str(paciente.get("genero", "M")).upper()
    genero   = {"M": 0, "F": 1}.get(genero_raw, 2)
    peso     = float(paciente.get("peso_kg", 70) or 70)
    altura   = float(paciente.get("altura_cm", 170) or 170)
    imc      = peso / ((altura / 100) ** 2) if altura > 0 else 25.0

    feats.extend([edad, genero, peso, altura, imc])

    # Foco de auscultación (importante clínicamente)
    foco_map = {
        "Aortico": 0, "Aórtico": 0,
        "Pulmonar": 1,
        "Tricuspideo": 2, "Tricuspídeo": 2,
        "Mitral": 3,
        "Apex": 4,
        "Otro": 5
    }
    diagnostico = metadata.get("diagnostico", {})
    foco = diagnostico.get("foco_auscultacion", "Otro")
    feats.append(float(foco_map.get(foco, 5)))

    # Código de hospital y consultorio como numérico
    ubicacion = metadata.get("ubicacion", {})
    cod_hospital = int(ubicacion.get("codigo_hospital", 0) or 0)
    feats.append(float(cod_hospital))

    return np.array(feats, dtype=np.float32)


def build_full_feature_vector(
    y_principal: np.ndarray,
    metadata: dict,
    y_ecg: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Construye el vector completo combinando:
    - Características del audio principal
    - Características del audio ECG (si existe)
    - Metadatos clínicos
    """
    feats_audio = extract_audio_features(y_principal)

    if y_ecg is not None:
        feats_ecg = extract_audio_features(y_ecg)
        # Diferencia energética entre señales (rasgo clínico)
        energy_diff = np.abs(feats_audio[:N_MFCC] - feats_ecg[:N_MFCC])
        combined_audio = np.concatenate([feats_audio, feats_ecg, energy_diff])
    else:
        combined_audio = feats_audio

    feats_meta = extract_metadata_features(metadata)
    full_vector = np.concatenate([combined_audio, feats_meta])

    return full_vector
