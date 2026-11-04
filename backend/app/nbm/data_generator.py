"""Genera un dataset sintetico realista para entrenar el NBM.

Escenario simulado cercano a un entorno universitario real:
- 500 usuarios ficticios con rol asignado (estudiante, profesor, directivo,
  administrador, auditor).
- Distribucion horaria realista (picos 8-12 y 14-18, baja madrugada).
- Fines de semana con menos actividad.
- ~1% de anomalias (fuerza bruta, escaneo, exfiltracion, nocturno,
  password spraying, movimiento lateral).
- IPs internas (192.168.x.x) mayoritarias, externas (10.x.x.x) minoritarias.
- Metricas correlacionadas (mas peticiones -> mas bytes, etc.).

El objetivo es producir un dataset lo suficientemente grande (~50.000 eventos)
para que el split train/test y la matriz de confusion sean estadisticamente
significativos, y que los patrones de comportamiento sean lo mas realista
posible sin depender de datos reales de la CUJAE.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.nbm import config

# ---------------------------------------------------------------------------
# Catalogo de entidades ficticias
# ---------------------------------------------------------------------------
N_USUARIOS = 500

# Distribucion de roles en una universidad (aproximada)
ROLES_PESOS = [
    ("estudiante", 0.75),
    ("profesor", 0.15),
    ("jefe_proceso", 0.05),
    ("directivo", 0.03),
    ("administrador", 0.01),
    ("auditor", 0.01),
]

COMPONENTES = [
    "dashboard", "api_auth", "api_indicadores", "api_export", "api_modelos",
]
PROTOCOLOS = ["HTTPS", "SQL", "LDAP"]
PROTOCOLOS_PESOS = [0.85, 0.10, 0.05]

# Perfiles de comportamiento por rol (media diaria de acciones, horario tipico)
PERFILES_ROL = {
    "estudiante":    {"acciones_dia": (5, 30),   "horas": (7, 22),  "bytes": (0.5, 10)},
    "profesor":      {"acciones_dia": (20, 80),  "horas": (7, 20),  "bytes": (2, 30)},
    "jefe_proceso":  {"acciones_dia": (30, 120), "horas": (8, 18),  "bytes": (5, 50)},
    "directivo":     {"acciones_dia": (40, 150), "horas": (8, 18),  "bytes": (5, 60)},
    "administrador": {"acciones_dia": (30, 100), "horas": (7, 19),  "bytes": (3, 40)},
    "auditor":       {"acciones_dia": (10, 50),  "horas": (8, 17),  "bytes": (2, 25)},
}


def _elegir_rol(rng: np.random.Generator) -> str:
    roles, pesos = zip(*ROLES_PESOS)
    return str(rng.choice(roles, p=pesos))


def _hora_realista(rng: np.random.Generator, rol: str) -> int:
    """Genera una hora con distribucion realista segun el rol.

    Picos 8-12 y 14-18; baja actividad de madrugada; profesores y directivos
    no suelen acceder de noche.
    """
    h_min, h_max = PERFILES_ROL[rol]["horas"]
    # 85% en horario laboral, 15% fuera (pero dentro del rango del rol)
    if rng.random() < 0.85:
        return int(rng.integers(max(8, h_min), min(18, h_max) + 1))
    return int(rng.integers(h_min, h_max + 1))


def _ip_interna(rng: np.random.Generator) -> str:
    return f"192.168.{rng.integers(1, 20)}.{rng.integers(10, 250)}"


def _ip_externa(rng: np.random.Generator) -> str:
    return f"10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 254)}"


def _generar_usuarios(rng: np.random.Generator) -> dict[str, dict]:
    """Genera el catalogo de usuarios con su rol y perfil."""
    usuarios = {}
    for i in range(N_USUARIOS):
        rol = _elegir_rol(rng)
        username = f"user_{i:04d}"
        perfil = PERFILES_ROL[rol]
        acciones_media = rng.integers(*perfil["acciones_dia"])
        usuarios[username] = {
            "rol": rol,
            "acciones_dia": int(acciones_media),
            "ip_habitual": _ip_interna(rng) if rng.random() < 0.9 else _ip_externa(rng),
            "bytes_medio": float(rng.uniform(*perfil["bytes"])),
        }
    return usuarios


# ---------------------------------------------------------------------------
# Generacion de eventos normales
# ---------------------------------------------------------------------------
def _evento_normal(
    rng: np.random.Generator, usuario: str, info: dict, dia: int, hora: int
) -> dict:
    """Genera un evento de comportamiento normal para un usuario."""
    rol = info["rol"]
    componente = str(rng.choice(COMPONENTES, p=[0.4, 0.15, 0.25, 0.15, 0.05]))
    ip = info["ip_habitual"]
    # 10% de las veces accede desde otra IP interna (movilidad dentro del campus)
    if rng.random() < 0.10:
        ip = _ip_interna(rng)

    bytes_base = info["bytes_medio"]
    num_peticiones = int(max(1, rng.poisson(5)))
    intentos = int(rng.binomial(3, 0.03))     # 3% de eventos con intento fallido
    errores_4xx = int(rng.binomial(10, 0.05))
    errores_5xx = int(rng.binomial(10, 0.01))
    accesos_sensibles = int(rng.binomial(2, 0.05))

    return {
        "usuario": usuario,
        "rol": rol,
        "ip_origen": ip,
        "hora": hora,
        "dia_semana": dia,
        "componente": componente,
        "duracion_sesion_min": round(float(rng.gamma(2, 5)), 2),      # ~10 min tipico
        "num_peticiones_hora": num_peticiones,
        "intentos_fallidos": intentos,
        "errores_4xx": errores_4xx,
        "errores_5xx": errores_5xx,
        "bytes_transferidos_mb": round(float(rng.gamma(2, bytes_base / 2)), 2),
        "accesos_rutas_sensibles": accesos_sensibles,
        "usuarios_distintos_ip": int(rng.integers(1, 3)),
        "es_horario_laboral": 1 if 8 <= hora < 19 else 0,
        "protocolo": str(rng.choice(PROTOCOLOS, p=PROTOCOLOS_PESOS)),
        "is_anomalous": 0,
    }


# ---------------------------------------------------------------------------
# Generacion de anomalias (6 tipos)
# ---------------------------------------------------------------------------
TIPOS_ANOMALIA = [
    "fuerza_bruta", "escaneo", "exfiltracion",
    "nocturno", "password_spray", "movimiento_lateral",
]


def _evento_anomalo(
    rng: np.random.Generator, usuario: str, info: dict, dia: int, hora: int
) -> dict:
    """Genera un evento anomalo alterando las features de uno normal."""
    tipo = str(rng.choice(TIPOS_ANOMALIA))
    base = _evento_normal(rng, usuario, info, dia, hora)
    base["is_anomalous"] = 1

    if tipo == "fuerza_bruta":
        base.update(
            intentos_fallidos=int(rng.integers(10, 50)),
            num_peticiones_hora=int(rng.integers(80, 400)),
            componente="api_auth",
        )
    elif tipo == "escaneo":
        base.update(
            errores_4xx=int(rng.integers(50, 400)),
            num_peticiones_hora=int(rng.integers(150, 800)),
            accesos_rutas_sensibles=int(rng.integers(5, 40)),
            componente="api_export",
        )
    elif tipo == "exfiltracion":
        base.update(
            bytes_transferidos_mb=round(float(rng.uniform(100, 1500)), 2),
            duracion_sesion_min=round(float(rng.uniform(60, 300)), 2),
            num_peticiones_hora=int(rng.integers(40, 200)),
            componente="api_export",
        )
    elif tipo == "nocturno":
        base.update(
            hora=int(rng.choice([0, 1, 2, 3, 4, 22, 23])),
            es_horario_laboral=0,
        )
    elif tipo == "password_spray":
        base.update(
            usuarios_distintos_ip=int(rng.integers(5, 25)),
            intentos_fallidos=int(rng.integers(10, 50)),
            componente="api_auth",
            ip_origen=_ip_externa(rng),
        )
    elif tipo == "movimiento_lateral":
        base.update(
            accesos_rutas_sensibles=int(rng.integers(10, 50)),
            usuarios_distintos_ip=int(rng.integers(3, 10)),
            componente="api_modelos",
            ip_origen=_ip_externa(rng),
        )
    return base


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------
def generar_dataset_sintetico(
    n_total: int = config.N_REGISTROS,
    proporcion_anomalias: float = config.PROPORCION_ANOMALIAS,
    seed: int = config.SEED,
    dias_simulados: int = 30,
) -> pd.DataFrame:
    """Genera un dataset sintetico realista.

    Args:
        n_total: numero total de eventos (~50.000 por defecto).
        proporcion_anomalias: fraccion de eventos anomalos (~1% por defecto).
        seed: semilla para reproducibilidad.
        dias_simulados: ventana temporal simulada (30 dias por defecto).

    Returns:
        DataFrame con features y etiqueta is_anomalous.
    """
    rng = np.random.default_rng(seed)
    usuarios = _generar_usuarios(rng)

    n_anom = int(n_total * proporcion_anomalias)
    n_norm = n_total - n_anom

    filas: list[dict] = []

    # Normales: distribuir uniformemente en los dias simulados
    lista_usuarios = list(usuarios.keys())
    for _ in range(n_norm):
        u = str(rng.choice(lista_usuarios))
        info = usuarios[u]
        dia = int(rng.integers(0, dias_simulados))
        dia_semana = dia % 7
        # Fines de semana: menos actividad
        if dia_semana >= 5 and rng.random() < 0.7:
            continue  # se descarta (menos eventos en fin de semana)
        hora = _hora_realista(rng, info["rol"])
        filas.append(_evento_normal(rng, u, info, dia_semana, hora))

    # Anomalias: distribuir aleatoriamente (pueden caer en cualquier dia/hora)
    for _ in range(n_anom):
        u = str(rng.choice(lista_usuarios))
        info = usuarios[u]
        dia = int(rng.integers(0, dias_simulados))
        hora = _hora_realista(rng, info["rol"])
        filas.append(_evento_anomalo(rng, u, info, dia % 7, hora))

    df = pd.DataFrame(filas)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def guardar_dataset(df: pd.DataFrame, ruta=config.DATASET_CSV) -> str:
    """Guarda el DataFrame en CSV y devuelve el hash SHA256."""
    df.to_csv(ruta, index=False)
    contenido = ruta.read_bytes()
    return hashlib.sha256(contenido).hexdigest()