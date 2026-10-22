"""Detector por umbrales estadísticos (puntuación Z).
Ventajas:
- Interpretable: cada alerta se puede explicar con "la métrica X se desvió
  N desviaciones estándar de la media".
- Ultrarrápido: O(n_features) por evento.
- No requiere entrenamiento iterativo (solo calcular media y std).

Limitaciones:
- Asume distribución aproximadamente normal (no siempre se cumple).
- No captura correlaciones entre features (los otros algoritmos sí).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.nbm import config
from app.nbm.algorithms import ResultadoAlgoritmo

# Features numéricas alineadas con la Tabla 1.2 del Informe PP2.
# Se listan por categoría (temporal, usuario, red, recurso, operacional, contextual).
# En esta iteración se incluyen las que el dataset sintético ya genera; las demás
# se documentan como trabajo futuro.
FEATURES_NUMERICAS = {
    # Temporal
    "hora": "hora_del_dia",           # 0-23
    "dia_semana": "dia_semana",       # 0-6
    "es_horario_laboral": "es_fin_semana/laboral",  # binaria
    # Usuario
    "duracion_sesion_min": "duracion_sesion",
    "num_peticiones_hora": "num_acciones",
    "accesos_rutas_sensibles": "recursos_accedidos",
    "intentos_fallidos": "intentos_fallidos",
    # Red
    "ip_externa": "ip_externa",  # derivada de ip_origen
    "usuarios_distintos_ip": "ip_diversidad",
    "errores_4xx": "error_rate_4xx",
    "errores_5xx": "error_rate_5xx",
    "bytes_transferidos_mb": "data_volume_mb",
}

UMBRAL_Z = 3.0  # |Z| > 3.0 -> anomalía (99.7% de la distribución normal)


@dataclass
class ModeloZ:
    """Estadísticas del modelo (media y std por feature)."""
    medias: np.ndarray
    desviaciones: np.ndarray
    feature_names: list[str]
    umbral: float


def entrenar_detector_z(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    umbral: float = UMBRAL_Z,
) -> ModeloZ:
    """Calcula media y desviación estándar de cada feature sobre datos normales."""
    X_normal = X_train[y_train == 0]
    medias = X_normal.mean(axis=0)
    desviaciones = X_normal.std(axis=0)
    # Evitar división por cero: donde std=0, poner 1.0 (feature constante)
    desviaciones = np.where(desviaciones == 0, 1.0, desviaciones)
    return ModeloZ(
        medias=medias,
        desviaciones=desviaciones,
        feature_names=feature_names or [f"f{i}" for i in range(X_train.shape[1])],
        umbral=umbral,
    )


def predecir_detector_z(modelo: ModeloZ, X_test: np.ndarray) -> np.ndarray:
    """Predice anomalías calculando |Z| máximo por evento.

    Un evento es anomalía si CUALQUIERA de sus features supera el umbral.
    """
    Z = np.abs((X_test - modelo.medias) / modelo.desviaciones)
    return (Z.max(axis=1) > modelo.umbral).astype(int)


def ejecutar_detector_z(X_train, y_train, X_test, feature_names=None) -> ResultadoAlgoritmo:
    """Interfaz compatible con los otros algoritmos del runner."""
    t0 = time.perf_counter()
    modelo = entrenar_detector_z(X_train, y_train, feature_names=feature_names)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = predecir_detector_z(modelo, X_test)
    t_pred = time.perf_counter() - t0

    # Guardar medias y desviaciones como "hiperparámetros" para trazabilidad
    hiperparams = {
        "umbral_z": modelo.umbral,
        "medias": {f: round(float(m), 4) for f, m in zip(modelo.feature_names, modelo.medias)},
        "desviaciones": {f: round(float(d), 4) for f, d in zip(modelo.feature_names, modelo.desviaciones)},
    }

    return ResultadoAlgoritmo(
        nombre="DetectorZ",
        hiperparametros=hiperparams,
        y_pred=y_pred,
        tiempo_entrenamiento_s=t_fit,
        tiempo_prediccion_s=t_pred,
        cpu_entrenamiento_pct=0.0,
        cpu_prediccion_pct=0.0,
        ram_entrenamiento_mb=0.0,
        ram_prediccion_mb=0.0,
        extra={"umbral_z": modelo.umbral},
    )