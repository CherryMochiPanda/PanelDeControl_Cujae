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
    "hora": "hora_del_dia",
    "dia_semana": "dia_semana",
    "es_horario_laboral": "es_fin_semana/laboral",
    # Usuario
    "duracion_sesion_min": "duracion_sesion",
    "num_peticiones_hora": "num_acciones",
    "accesos_rutas_sensibles": "recursos_accedidos",
    "intentos_fallidos": "intentos_fallidos",
    # Red
    "ip_externa": "ip_externa",
    "usuarios_distintos_ip": "ip_diversidad",
    "errores_4xx": "error_rate_4xx",
    "errores_5xx": "error_rate_5xx",
    "bytes_transferidos_mb": "data_volume_mb",
    # Categoricas one-hot (informativas para el detector)
    "componente_api_auth": "componente_auth",
    "componente_api_export": "componente_export",
    "componente_api_indicadores": "componente_indicadores",
    "componente_api_modelos": "componente_modelos",
    "componente_dashboard": "componente_dashboard",
    "rol_admin": "rol_admin",
    "rol_auditor": "rol_auditor",
    "rol_directivo": "rol_directivo",
    "rol_estudiante": "rol_estudiante",
    "rol_jefe_proceso": "rol_jefe_proceso",
    "rol_profesor": "rol_profesor",
    "protocolo_HTTPS": "protocolo_https",
    "protocolo_LDAP": "protocolo_ldap",
    "protocolo_SQL": "protocolo_sql",
}

UMBRAL_Z = 3.0  # |Z| > 3.0 -> anomalía (99.7% de la distribución normal)


@dataclass
class ModeloZ:
    medias: np.ndarray
    desviaciones: np.ndarray
    umbrales: np.ndarray  # umbral por feature (percentil 99 de |Z| en train)
    feature_names: list[str]
    umbral_global: float  # umbral global (percentil 99 del max |Z| por evento)

def entrenar_detector_z(X_train, y_train, *, feature_names=None, percentil=99):
    X_normal = X_train[y_train == 0]
    medias = X_normal.mean(axis=0)
    desviaciones = X_normal.std(axis=0)
    desviaciones = np.where(desviaciones == 0, 1.0, desviaciones)
    Z_normal = np.abs((X_normal - medias) / desviaciones)
    # Umbral global: percentil 99 del max |Z| por evento normal
    max_z_normal = Z_normal.max(axis=1)
    umbral_global = float(np.percentile(max_z_normal, percentil))
    return ModeloZ(
        medias=medias, desviaciones=desviaciones,
        umbrales=np.full(medias.shape, umbral_global),
        feature_names=feature_names or [f"f{i}" for i in range(X_train.shape[1])],
        umbral_global=umbral_global,
    )

def predecir_detector_z(modelo, X_test):
    Z = np.abs((X_test - modelo.medias) / modelo.desviaciones)
    return (Z.max(axis=1) > modelo.umbral_global).astype(int)


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