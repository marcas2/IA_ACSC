"""
utils/local_server.py
Conector con el servidor local Apache en 172.16.10.200:5002
Maneja:
  - Descarga de audios para entrenamiento
  - Subida de nuevos audios (vía PUT/POST o escritura directa si hay acceso)
  - Listado de archivos disponibles
"""
import io
import json
import logging
from typing import Optional, List, Dict
from pathlib import Path

import requests
import httpx

from config import LOCAL_SERVER_BASE_URL, REMOTE_FOLDERS

logger = logging.getLogger(__name__)

# Timeout generoso para transferencias de audio
TIMEOUT = httpx.Timeout(30.0, connect=10.0)


# ──────────────────────────────────────────────────────────────────────────────
# DESCARGA
# ──────────────────────────────────────────────────────────────────────────────

def download_file(folder: str, filename: str) -> Optional[bytes]:
    """
    Descarga un archivo desde el servidor Apache local.
    URL resultante: http://172.16.10.200:5002/{folder}/{filename}
    """
    url = f"{LOCAL_SERVER_BASE_URL}/{folder}/{filename}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        logger.info(f"Descargado: {url} ({len(resp.content)} bytes)")
        return resp.content
    except requests.RequestException as e:
        logger.warning(f"No se pudo descargar {url}: {e}")
        return None


def download_audio_principal(filename: str) -> Optional[bytes]:
    return download_file(REMOTE_FOLDERS["audio_principal"], filename)


def download_audio_json(filename: str) -> Optional[dict]:
    """Descarga y parsea un JSON clínico desde el servidor local."""
    raw = download_file(REMOTE_FOLDERS["audio_json"], filename)
    if raw:
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"JSON inválido en {filename}: {e}")
    return None


def download_audio_ecg(filename: str) -> Optional[bytes]:
    return download_file(REMOTE_FOLDERS["audio_ecg"], filename)


def download_audio_ecg1(filename: str) -> Optional[bytes]:
    return download_file(REMOTE_FOLDERS["audio_ecg_1"], filename)


def download_audio_ecg2(filename: str) -> Optional[bytes]:
    return download_file(REMOTE_FOLDERS["audio_ecg_2"], filename)


# ──────────────────────────────────────────────────────────────────────────────
# LISTADO (Apache Directory Listing parser simple)
# ──────────────────────────────────────────────────────────────────────────────

def list_files_in_folder(folder: str) -> List[str]:
    """
    Parsea el directory listing HTML de Apache para obtener los nombres
    de archivos en una carpeta del servidor local.
    """
    url = f"{LOCAL_SERVER_BASE_URL}/{folder}/"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        # Extraer hrefs de la página de listado
        from html.parser import HTMLParser

        class LinkParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links: List[str] = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    for attr, val in attrs:
                        if attr == "href" and val and not val.startswith("?") \
                                and not val.startswith("/") and val != "../":
                            self.links.append(val)

        parser = LinkParser()
        parser.feed(resp.text)
        files = [lnk for lnk in parser.links if not lnk.endswith("/")]
        logger.info(f"Carpeta {folder}: {len(files)} archivos encontrados")
        return files
    except requests.RequestException as e:
        logger.warning(f"No se pudo listar {url}: {e}")
        return []


def list_all_available_samples() -> List[Dict]:
    """
    Construye la lista de muestras disponibles emparejando JSON ↔ audio principal.
    Retorna lista de dicts con claves: json_file, wav_file, ecg_file, ecg1_file, ecg2_file
    """
    json_files = list_files_in_folder(REMOTE_FOLDERS["audio_json"])
    wav_files  = set(list_files_in_folder(REMOTE_FOLDERS["audio_principal"]))
    ecg_files  = set(list_files_in_folder(REMOTE_FOLDERS["audio_ecg"]))
    ecg1_files = set(list_files_in_folder(REMOTE_FOLDERS["audio_ecg_1"]))
    ecg2_files = set(list_files_in_folder(REMOTE_FOLDERS["audio_ecg_2"]))

    samples = []
    for jf in json_files:
        if not jf.endswith(".json"):
            continue
        base = jf.replace(".json", "")
        wav   = base + ".wav"
        ecg   = base + "_ECG.wav"
        ecg1  = base + "_ECG_1.wav"
        ecg2  = base + "_ECG_2.wav"

        if wav not in wav_files:
            logger.warning(f"Audio principal no encontrado para {jf}, omitiendo")
            continue

        samples.append({
            "json_file": jf,
            "wav_file":  wav,
            "ecg_file":  ecg  if ecg  in ecg_files  else None,
            "ecg1_file": ecg1 if ecg1 in ecg1_files else None,
            "ecg2_file": ecg2 if ecg2 in ecg2_files else None,
        })

    logger.info(f"Total muestras emparejadas: {len(samples)}")
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# SUBIDA  (requiere que Apache tenga WebDAV o un endpoint PUT habilitado)
# Si el servidor solo tiene directory listing, esta función envía via PUT.
# Ajusta el método según la configuración de tu Apache.
# ──────────────────────────────────────────────────────────────────────────────

def upload_file(folder: str, filename: str, data: bytes,
                content_type: str = "application/octet-stream") -> bool:
    """
    Sube un archivo al servidor local.
    Usa HTTP PUT (requiere WebDAV o módulo dav en Apache).
    Si tu servidor usa otro método, ajusta aquí.
    """
    url = f"{LOCAL_SERVER_BASE_URL}/{folder}/{filename}"
    try:
        resp = requests.put(
            url,
            data=data,
            headers={"Content-Type": content_type},
            timeout=30
        )
        if resp.status_code in (200, 201, 204):
            logger.info(f"Subido exitosamente: {url}")
            return True
        else:
            logger.error(f"Error subiendo {url}: HTTP {resp.status_code} - {resp.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"Excepción subiendo {url}: {e}")
        return False


def upload_training_sample(
    json_filename: str, json_data: bytes,
    wav_principal: bytes, wav_principal_name: str,
    wav_ecg: Optional[bytes] = None, wav_ecg_name: Optional[str] = None,
    wav_ecg1: Optional[bytes] = None, wav_ecg1_name: Optional[str] = None,
    wav_ecg2: Optional[bytes] = None, wav_ecg2_name: Optional[str] = None,
) -> Dict[str, bool]:
    """
    Sube todos los archivos de una muestra de entrenamiento al servidor local
    en sus carpetas correspondientes.
    """
    results = {}

    results["json"] = upload_file(
        REMOTE_FOLDERS["audio_json"], json_filename, json_data, "application/json"
    )
    results["audio_principal"] = upload_file(
        REMOTE_FOLDERS["audio_principal"], wav_principal_name, wav_principal, "audio/wav"
    )
    if wav_ecg and wav_ecg_name:
        results["ecg"] = upload_file(
            REMOTE_FOLDERS["audio_ecg"], wav_ecg_name, wav_ecg, "audio/wav"
        )
    if wav_ecg1 and wav_ecg1_name:
        results["ecg_1"] = upload_file(
            REMOTE_FOLDERS["audio_ecg_1"], wav_ecg1_name, wav_ecg1, "audio/wav"
        )
    if wav_ecg2 and wav_ecg2_name:
        results["ecg_2"] = upload_file(
            REMOTE_FOLDERS["audio_ecg_2"], wav_ecg2_name, wav_ecg2, "audio/wav"
        )

    return results
