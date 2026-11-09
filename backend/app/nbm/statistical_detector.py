"""Detector por umbrales estadisticos (puntuacion Z adaptativa).

Implementa la "Capa 1: Deteccion estadistica rapida" del NBM. Calcula media
y desviacion estandar de cada feature numerica sobre los datos normales de
entrenamiento, y marca como anomalia cualquier evento cuyo |Z| maximo supere
un umbral calibrado por percentil (en vez de un 3.0 fijo, que no se adapta a
las distintas distribuciones de las features).

Features: usa TODAS las features numericas del espacio codificado (incluidas
las one-hot de rol, componente y protocolo), porque tambien aportan informacion
(por ejemplo, componente_api_auth es clave para detectar fuerza bruta).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.nbm import config
from app.nbm.algorithms import ResultadoAlgoritmo

UMBRAL_PERCENTIL = 99.0  # percentil del max |Z| en datos normales


@dataclass
class ModeloZ:
    """Estadisticas del modelo Z."""
    medias: np.ndarray
    desviaciones: np.ndarray
    feature_names: list[str]
    umbral_global: float


def entrenar_detector_z(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    percentil: float = UMBRAL_PERCENTIL,
) -> ModeloZ:
    """Calcula media, std y umbral global (percentil del max |Z| en normales)."""
    X_normal = X_train[y_train == 0]
    medias = X_normal.mean(axis=0)
    desviaciones = X_normal.std(axis=0)
    desviaciones = np.where(desviaciones == 0, 1.0, desviaciones)

    Z_normal = np.abs((X_normal - medias) / desviaciones)
    max_z_normal = Z_normal.max(axis=1)
    umbral_global = float(np.percentile(max_z_normal, percentil))

    return ModeloZ(
        medias=medias,
        desviaciones=desviaciones,
        feature_names=feature_names or [f"f{i}" for i in range(X_train.shape[1])],
        umbral_global=umbral_global,
    )


def predecir_detector_z(modelo: ModeloZ, X_test: np.ndarray) -> np.ndarray:
    """Predice anomalias: |Z| maximo por evento > umbral_global."""
    Z = np.abs((X_test - modelo.medias) / modelo.desviaciones)
    return (Z.max(axis=1) > modelo.umbral_global).astype(int)


def ejecutar_detector_z(X_train, y_train, X_test, feature_names=None) -> ResultadoAlgoritmo:
    """Interfaz compatible con el runner."""
    t0 = time.perf_counter()
    modelo = entrenar_detector_z(X_train, y_train, feature_names=feature_names)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = predecir_detector_z(modelo, X_test)
    t_pred = time.perf_counter() - t0

    hiperparams = {
        "umbral_global": round(modelo.umbral_global, 4),
        "percentil": UMBRAL_PERCENTIL,
        "n_features": len(modelo.feature_names),
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
        extra={"umbral_global": modelo.umbral_global},
    )