# 🫀 ValvIA — Sistema de Detección de Valvulopatías

Sistema de Inteligencia Artificial para detectar valvulopatías cardiacas a partir
de **fonocardiogramas (sonidos cardiacos WAV)** y **metadatos clínicos (JSON)**.

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    NUBE (ValvIA API)                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FastAPI - Puerto 8000                               │   │
│  │                                                      │   │
│  │  POST /api/v1/train/full    ← SERVICIO 1             │   │
│  │  POST /api/v1/train/sample  ← SERVICIO 2             │   │
│  │  POST /api/v1/diagnose      ← SERVICIO 3             │   │
│  │  GET  /api/v1/metrics       ← Evaluación             │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↕ HTTP                             │
└─────────────────────────────────────────────────────────────┘
                           ↕ Red interna / VPN
┌─────────────────────────────────────────────────────────────┐
│              SERVIDOR LOCAL (172.16.10.200:5002)             │
│                                                             │
│  Apache HTTP Server                                         │
│  ├── Audios/        ← audio_principal (.wav)                │
│  ├── audios-json/   ← metadatos clínicos (.json)            │
│  ├── ECG/           ← audio ECG (.wav)                      │
│  ├── ECG_1/         ← audio ECG_1 (.wav)                    │
│  └── ECG_2/         ← audio ECG_2 (.wav)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Modelo de IA

### Arquitectura
- **Ensemble de votación suave** (≥30 muestras):
  - Gradient Boosting Classifier (peso 2)
  - Random Forest Classifier (peso 2)
  - SVM Calibrado con kernel RBF (peso 1)
- **SGD Classifier** (<30 muestras o aprendizaje incremental)
- **StandardScaler** para normalización de features

### Características Extraídas (~238 por muestra)
| Grupo | Características |
|-------|----------------|
| MFCC (mean + std) | 80 features |
| Delta MFCC (mean) | 40 features |
| Mel-Spectrogram | 64 features |
| Chroma | 12 features |
| Spectral (centroid, bandwidth, rolloff, ZCR, RMS) | 7 features |
| Metadatos clínicos (edad, género, peso, IMC, foco) | 7 features |

### Clases
- `Normal` — Sin valvulopatía detectable
- `Anomalia` — Valvulopatía presente

---

## 🚀 Instalación y Despliegue

### Con Docker (recomendado)

```bash
# 1. Clonar / copiar el proyecto
cp .env.example .env
# Editar .env si es necesario

# 2. Construir e iniciar
docker-compose up -d --build

# 3. Verificar
curl http://localhost:8000/health
```

### Sin Docker

```bash
# Requisitos: Python 3.10+
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env

# Iniciar
python main.py
```

---

## 📡 Endpoints de la API

### 🔵 SERVICIO 1 — Entrenamiento completo
```
POST /api/v1/train/full
```
Descarga **todos** los audios y JSON del servidor local y entrena el modelo desde cero.

**Respuesta:**
```json
{
  "status": "success",
  "dataset": {
    "total_muestras": 120,
    "normales": 95,
    "anomalias": 25,
    "n_features": 238
  },
  "train_metrics": {
    "n_samples": 120,
    "train_accuracy": 0.9583
  },
  "cross_validation": {
    "cv_folds": 5,
    "accuracy_mean": 0.8750,
    "accuracy_std": 0.0412,
    "f1_mean": 0.8634,
    "roc_auc_mean": 0.9120
  }
}
```

---

### 🟢 SERVICIO 2 — Nueva muestra de entrenamiento
```
POST /api/v1/train/sample
Content-Type: multipart/form-data
```

**Form fields:**
| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `json_metadata` | string (JSON) | ✅ | Metadatos clínicos con `diagnostico.estado` |
| `audio_principal` | file (.wav) | ✅ | Fonocardiograma principal |
| `audio_ecg` | file (.wav) | ❌ | Audio ECG |
| `audio_ecg_1` | file (.wav) | ❌ | Audio ECG_1 |
| `audio_ecg_2` | file (.wav) | ❌ | Audio ECG_2 |

El JSON **debe** incluir `diagnostico.estado` para usarse como etiqueta.

**Ejemplo con curl:**
```bash
curl -X POST http://localhost:8000/api/v1/train/sample \
  -F "json_metadata={\"metadata\":{\"edad\":45},\"paciente\":{\"genero\":\"M\",\"peso_kg\":80,\"altura_cm\":175},\"diagnostico\":{\"estado\":\"Normal\",\"foco_auscultacion\":\"Mitral\",\"categoria_anomalia\":null}}" \
  -F "audio_principal=@SC_20260314_0001.wav" \
  -F "audio_ecg=@SC_20260314_0001_ECG.wav"
```

---

### 🔴 SERVICIO 3 — Diagnóstico
```
POST /api/v1/diagnose
Content-Type: multipart/form-data
```

**Form fields:**
| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `json_metadata` | string (JSON) | ✅ | Metadatos del paciente (sin diagnóstico) |
| `audio` | file (.wav) | ✅ | Fonocardiograma a analizar |

**Respuesta:**
```json
{
  "timestamp_diagnostico": "2026-04-09T14:30:00",
  "archivo_analizado": "SC_20260409_0001.wav",
  "paciente": {
    "edad": 55,
    "genero": "F",
    "peso_kg": 65.0,
    "altura_cm": 160.0
  },
  "foco_auscultacion": "Aortico",
  "resultado_ia": {
    "diagnostico": "Anomalia",
    "tiene_valvulopatia": true,
    "probabilidad_anomalia": 0.8734,
    "probabilidad_normal": 0.1266,
    "confianza": "Alta",
    "modelo_entrenado_con": 120
  },
  "recomendacion": "⚠️ Alta probabilidad de valvulopatía detectada. Se recomienda evaluación cardiológica urgente con ecocardiograma Doppler.",
  "almacenamiento": {
    "audio_guardado": true,
    "json_guardado": true
  }
}
```

---

### 📊 Evaluación del modelo
```
GET /api/v1/metrics
```
Retorna accuracy, F1, ROC-AUC, matriz de confusión y validación cruzada.

```
GET /api/v1/metrics/history
```
Historial completo de evaluaciones.

---

## 🔒 Consideraciones Legales y de Privacidad

1. **Los audios cardiacos NUNCA salen del servidor local** (172.16.10.200:5002).
   La nube solo recibe los **vectores de características** extraídos, no los audios.
   
   > ⚠️ En el Servicio 2 y 3, los audios se reciben temporalmente en la nube solo
   > para extracción de características y luego se envían al servidor local.
   > Considerar implementar extracción de características en el cliente para
   > máxima privacidad.

2. **El modelo se persiste en la nube** (solo vectores numéricos, sin datos de pacientes).

3. **Los JSON clínicos** también se almacenan en el servidor local.

---

## 🔧 Configuración del Servidor Apache (172.16.10.200:5002)

Para habilitar la subida de archivos (Servicio 2 y 3), habilita WebDAV en Apache:

```apache
# En /etc/apache2/sites-available/valvia.conf
<VirtualHost *:5002>
    ServerName 172.16.10.200
    DocumentRoot /var/www/valvia

    # Habilitar WebDAV para PUT
    <Directory /var/www/valvia>
        DAV On
        Options +Indexes
        AllowOverride None
        Require all granted
    </Directory>

    LoadModule dav_module modules/mod_dav.so
    LoadModule dav_fs_module modules/mod_dav_fs.so
</VirtualHost>
```

```bash
sudo a2enmod dav dav_fs
sudo systemctl restart apache2
```

---

## 📁 Estructura del Proyecto

```
valvulopatias/
├── main.py                          # API FastAPI principal
├── config.py                        # Configuración central
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── models/
│   ├── __init__.py
│   └── valvulopatia_model.py        # Modelo IA (Ensemble + SGD)
├── services/
│   ├── __init__.py
│   ├── train_service.py             # Servicio 1: entrenamiento completo
│   ├── new_sample_service.py        # Servicio 2: muestra incremental
│   ├── diagnosis_service.py         # Servicio 3: diagnóstico
│   └── metrics_service.py           # Evaluación y métricas
├── utils/
│   ├── __init__.py
│   ├── feature_extractor.py         # Extracción de características de audio
│   └── local_server.py              # Conector Apache servidor local
└── data/
    ├── metrics_history.json         # Historial de métricas
    ├── training_log.json            # Log de entrenamientos
    └── reports/                     # Gráficas generadas
```
