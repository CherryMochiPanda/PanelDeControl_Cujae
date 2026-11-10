"""Servicio de inferencia: transforma eventos y predice con OneClassSVM.

Recibe eventos crudos (dicts o EventoIn), los convierte al vector de features
con el MISMO orden y encoding que el entrenamiento, aplica el scaler y predice
con el modelo cargado.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.nbm import model_store

log = logging.getLogger("nbm.detection")


def _evento_a_features(evento: dict[str, Any], feature_names: list[str]) -> np.ndarray:
    """Convierte un evento crudo al vector de features con el orden del modelo.

    Las features categoricas (rol, componente, protocolo) se convierten en
    one-hot. Si el valor no coincide con ninguna columna conocida, todas las
    one-hot de esa categoria quedan en 0 (equivalente a categoria desconocida).
    """
    vec = {f: 0.0 for f in feature_names}

    # Numericas
    for k in [
        "hora", "dia_semana", "es_horario_laboral",
        "duracion_sesion_min", "num_peticiones_hora",
        "intentos_fallidos", "errores_4xx", "errores_5xx",
        "bytes_transferidos_mb", "accesos_rutas_sensibles",
        "usuarios_distintos_ip", "ip_externa",
    ]:
        if k in vec:
            vec[k] = float(evento.get(k, 0))

    # Categoricas -> one-hot
    for k in ["rol", "componente", "protocolo"]:
        col = f"{k}_{evento.get(k, '')}"
        if col in vec:
            vec[col] = 1.0

    return np.array([vec[f] for f in feature_names], dtype=np.float64)


def detectar_eventos(eventos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detecta anomalias en un lote de eventos.

    Args:
        eventos: lista de dicts con las features crudas del evento.

    Returns:
        Lista de dicts con {is_anomaly, score, index}.

    Raises:
        RuntimeError: si no hay modelo entrenado disponible.
    """
    artefacto = model_store.load_model()
    if artefacto is None:
        raise RuntimeError(
            "No hay modelo entrenado. Ejecuta: python -m app.nbm.runner --save-model"
        )

    model = artefacto["model"]
    scaler = artefacto["scaler"]
    features = artefacto["features"]

    X = np.vstack([_evento_a_features(e, features) for e in eventos])
    X_s = scaler.transform(X)

    # OC-SVM: -1 = anomalia, +1 = normal. decision_function: mas bajo = mas anomalo.
    raw = model.predict(X_s)
    scores = model.decision_function(X_s)

    return [
        {
            "index": i,
            "is_anomaly": bool(raw[i] == -1),
            "score": float(scores[i]),
        }
        for i in range(len(eventos))
    ]