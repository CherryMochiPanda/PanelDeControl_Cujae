"""Genera un dataset sintetico realista para entrenar el NBM.

Las features reflejan las metricas de comportamiento del informe PP2
(Tabla 2.10) y el esquema propuesto en el documento "Normal Behavior Model".

Incluye IPs y usernames ficticios para facilitar ejemplos en la presentacion.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.nbm import config

USUARIOS = [
    "jperez", "mgarcia", "alopez", "rfernandez", "dcastillo",
    "lmartinez", "pramirez", "scastro", "itorres", "vsanchez",
    "admin", "auditor", "directivo1", "jefe_proc", "analista_bi",
]
ROLES = ["directivo", "jefe_proceso", "auditor", "administrador", "alta_direccion"]
COMPONENTES = ["dashboard", "api_auth", "api_indicadores", "api_export", "api_modelos"]
RUTAS_SENSIBLES = ["/admin", "/api/auth", "/api/models", "/api/users"]
PROTOCOLOS = ["HTTPS", "SQL", "LDAP"]


def _generar_normales(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Genera n filas de comportamiento normal (horario laboral, poco volumen)."""
    horas = rng.integers(8, 19, size=n)  # 8am-6pm
    usuarios = rng.choice(USUARIOS, size=n)
    roles = rng.choice(ROLES, size=n)
    componentes = rng.choice(COMPONENTES, size=n)
    ips = [f"192.168.{rng.integers(1, 5)}.{rng.integers(10, 250)}" for _ in range(n)]

    return pd.DataFrame({
        "usuario": usuarios,
        "rol": roles,
        "ip_origen": ips,
        "hora": horas,
        "dia_semana": rng.integers(0, 5, size=n),  # lun-vie
        "componente": componentes,
        "duracion_sesion_min": np.clip(rng.normal(30, 10, n), 1, 120).round(2),
        "num_peticiones_hora": np.clip(rng.normal(20, 10, n), 1, 100).astype(int),
        "intentos_fallidos": rng.binomial(5, 0.05, n),
        "errores_4xx": rng.binomial(20, 0.05, n),
        "errores_5xx": rng.binomial(10, 0.01, n),
        "bytes_transferidos_mb": np.clip(rng.normal(5, 3, n), 0.1, 50).round(2),
        "accesos_rutas_sensibles": rng.binomial(3, 0.03, n),
        "usuarios_distintos_ip": rng.integers(1, 2, size=n),
        "es_horario_laboral": 1,
        "protocolo": rng.choice(PROTOCOLOS, size=n, p=[0.8, 0.15, 0.05]),
        "is_anomalous": 0,
    })


def _generar_anomalias(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Genera n filas anomalas: fuerza bruta, escaneo, exfiltracion, nocturno."""
    tipos = rng.choice(["fuerza_bruta", "escaneo", "exfiltracion", "nocturno", "password_spray"], size=n)
    filas = []
    for tipo in tipos:
        usuario = str(rng.choice(USUARIOS))
        rol = str(rng.choice(ROLES))
        componente = str(rng.choice(COMPONENTES))
        ip = f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"

        base = {
            "usuario": usuario, "rol": rol, "ip_origen": ip,
            "hora": int(rng.integers(8, 19)),
            "dia_semana": int(rng.integers(0, 5)),
            "componente": componente,
            "duracion_sesion_min": float(np.clip(rng.normal(30, 10), 1, 120)),
            "num_peticiones_hora": int(np.clip(rng.normal(20, 10), 1, 100)),
            "intentos_fallidos": int(rng.integers(0, 2)),
            "errores_4xx": int(rng.integers(0, 3)),
            "errores_5xx": int(rng.integers(0, 2)),
            "bytes_transferidos_mb": float(np.clip(rng.normal(5, 3), 0.1, 50)),
            "accesos_rutas_sensibles": int(rng.integers(0, 2)),
            "usuarios_distintos_ip": int(rng.integers(1, 2)),
            "es_horario_laboral": 1,
            "protocolo": str(rng.choice(PROTOCOLOS)),
            "is_anomalous": 1,
        }

        if tipo == "fuerza_bruta":
            base.update(intentos_fallidos=int(rng.integers(10, 50)),
                        num_peticiones_hora=int(rng.integers(80, 300)),
                        componente="api_auth")
        elif tipo == "escaneo":
            base.update(errores_4xx=int(rng.integers(50, 300)),
                        num_peticiones_hora=int(rng.integers(150, 600)),
                        accesos_rutas_sensibles=int(rng.integers(5, 30)))
        elif tipo == "exfiltracion":
            base.update(bytes_transferidos_mb=float(rng.uniform(100, 800)),
                        duracion_sesion_min=float(rng.uniform(60, 240)),
                        num_peticiones_hora=int(rng.integers(50, 200)))
        elif tipo == "nocturno":
            base.update(hora=int(rng.choice([0, 1, 2, 3, 4, 22, 23])),
                        es_horario_laboral=0)
        elif tipo == "password_spray":
            base.update(usuarios_distintos_ip=int(rng.integers(5, 20)),
                        intentos_fallidos=int(rng.integers(10, 40)),
                        componente="api_auth")
        filas.append(base)

    return pd.DataFrame(filas)


def generar_dataset_sintetico(
    n_total: int = config.N_REGISTROS,
    proporcion_anomalias: float = config.PROPORCION_ANOMALIAS,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Genera el dataset sintetico y devuelve el DataFrame completo."""
    rng = np.random.default_rng(seed)
    n_anom = int(n_total * proporcion_anomalias)
    n_norm = n_total - n_anom

    df_norm = _generar_normales(n_norm, rng)
    df_anom = _generar_anomalias(n_anom, rng)
    df = pd.concat([df_norm, df_anom], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def guardar_dataset(df: pd.DataFrame, ruta=config.DATASET_CSV) -> str:
    """Guarda el DataFrame en CSV y devuelve el hash SHA256."""
    df.to_csv(ruta, index=False)
    contenido = ruta.read_bytes()
    return hashlib.sha256(contenido).hexdigest()