"""Persistencia y cache del modelo entrenado (OneClassSVM).

Guarda en disco un artefacto con:
- model: OneClassSVM entrenado
- scaler: StandardScaler ajustado al train
- features: lista ordenada de nombres de columnas
- metadata: metricas, hiperparametros, fecha de entrenamiento

Carga en memoria (cache) para evitar leer disco en cada request.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from app.nbm import config

log = logging.getLogger("nbm.model_store")

_lock = threading.Lock()
_cache: dict[str, Any] | None = None


def save_model(
    model: Any,
    scaler: Any,
    features: list[str],
    metadata: dict[str, Any],
    *,
    path: Path = config.MODEL_PATH,
) -> Path:
    """Guarda el artefacto del modelo en disco. Invalida la cache."""
    global _cache
    artefacto = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "metadata": {
            **metadata,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    with _lock:
        joblib.dump(artefacto, path)
        _cache = None  # invalidar cache para forzar recarga
    log.info("Modelo guardado en %s", path)
    return path


def load_model(path: Path = config.MODEL_PATH) -> dict[str, Any] | None:
    """Carga el artefacto del modelo (con cache). Devuelve None si no existe."""
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        if not path.exists():
            log.warning("No hay modelo entrenado en %s", path)
            return None
        _cache = joblib.load(path)
        log.info("Modelo cargado desde %s", path)
        return _cache


def get_metadata(path: Path = config.MODEL_PATH) -> dict[str, Any] | None:
    """Devuelve solo la metadata del modelo (para GET /modelo/info)."""
    artefacto = load_model(path)
    if artefacto is None:
        return None
    return artefacto.get("metadata", {})


def invalidate_cache() -> None:
    """Fuerza recarga en el proximo request (util tras reentrenar)."""
    global _cache
    with _lock:
        _cache = None