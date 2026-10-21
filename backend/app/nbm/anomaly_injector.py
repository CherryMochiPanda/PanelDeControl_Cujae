"""Inyeccion de anomalias controladas sobre un dataset real.

Dado un DataFrame con is_anomalous=0 (todos normales), inyecta N anomalias
de distintos tipos (fuerza bruta, escaneo, exfiltracion, nocturno, password
spray) modificando las features de filas seleccionadas al azar.

Esto permite calcular metricas supervisadas (matriz de confusion, F1, FPR)
sobre logs reales, igual que con el dataset sintetico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.nbm import config


def inyectar_anomalias(
    df: pd.DataFrame,
    *,
    proporcion: float = config.PROPORCION_ANOMALIAS,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Inyecta anomalias en un DataFrame normal (is_anomalous=0).

    Args:
        df: DataFrame con features reales, todos normales.
        proporcion: fraccion de filas a convertir en anomalia.
        seed: semilla para reproducibilidad.

    Returns:
        DataFrame con la misma cantidad de filas, algunas marcadas is_anomalous=1
        y con features alteradas para simular el ataque.
    """
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    df["is_anomalous"] = 0

    n_anom = max(1, int(len(df) * proporcion))
    indices = rng.choice(len(df), size=n_anom, replace=False)

    tipos = rng.choice(
        ["fuerza_bruta", "escaneo", "exfiltracion", "nocturno", "password_spray"],
        size=n_anom,
    )

    for idx, tipo in zip(indices, tipos):
        if tipo == "fuerza_bruta":
            df.at[idx, "intentos_fallidos"] = int(rng.integers(10, 50))
            df.at[idx, "num_peticiones_hora"] = int(rng.integers(80, 300))
            df.at[idx, "componente"] = "api_auth"
        elif tipo == "escaneo":
            df.at[idx, "errores_4xx"] = int(rng.integers(50, 300))
            df.at[idx, "num_peticiones_hora"] = int(rng.integers(150, 600))
            df.at[idx, "accesos_rutas_sensibles"] = int(rng.integers(5, 30))
        elif tipo == "exfiltracion":
            df.at[idx, "bytes_transferidos_mb"] = float(rng.uniform(100, 800))
            df.at[idx, "duracion_sesion_min"] = float(rng.uniform(60, 240))
        elif tipo == "nocturno":
            df.at[idx, "hora"] = int(rng.choice([0, 1, 2, 3, 4, 22, 23]))
            df.at[idx, "es_horario_laboral"] = 0
        elif tipo == "password_spray":
            df.at[idx, "usuarios_distintos_ip"] = int(rng.integers(5, 20))
            df.at[idx, "intentos_fallidos"] = int(rng.integers(10, 40))
            df.at[idx, "componente"] = "api_auth"

        df.at[idx, "is_anomalous"] = 1

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)