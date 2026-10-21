"""Extraccion de features desde logs reales (auditoria.security_logs).

Lee los logs JSONB de PostgreSQL y construye un DataFrame con el mismo
esquema que el dataset sintetico (data_generator.py), para que el runner
pueda entrenar con datos reales sin cambiar los algoritmos.

Como los logs reales no tienen etiqueta 'is_anomalous', el extractor
devuelve is_anomalous=0 por defecto. La inyeccion de anomalias se hace
en un paso posterior (anomaly_injector.py) para poder calcular metricas
supervisadas (matriz de confusion, F1, etc.).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.auditoria import SecurityLog
from app.nbm import config

# Mapeo de resource_type del log real a componente del dataset
RESOURCE_A_COMPONENTE = {
    "usuario": "api_auth",
    "configuracion": "api_modelos",
    "indicador": "api_indicadores",
    "fuente": "api_export",
    "proceso": "api_indicadores",
    "curso": "api_indicadores",
}


def _ip_externa(ip: str) -> int:
    """1 si la IP no es de red interna CUJAE (192.168.x.x), 0 si lo es."""
    if not ip:
        return 0
    return 0 if ip.startswith("192.168.") else 1


def _hora_laboral(hora: int) -> int:
    return 1 if 8 <= hora < 19 else 0


async def extraer_features_reales(
    session: AsyncSession,
    *,
    desde: datetime | None = None,
    hasta: datetime | None = None,
    limite: int = 50000,
) -> pd.DataFrame:
    """Extrae features agregadas desde auditoria.security_logs.

    Agrupa por (usuario, hora, componente) y calcula metricas de comportamiento
    equivalentes a las del dataset sintetico.

    Args:
        session: sesion async.
        desde: fecha inicio (default: hace 30 dias).
        hasta: fecha fin (default: ahora).
        limite: maximo de filas a leer (para no saturar memoria).

    Returns:
        DataFrame con columnas equivalentes a data_generator.
    """
    if hasta is None:
        hasta = datetime.now(timezone.utc)
    if desde is None:
        desde = hasta - timedelta(days=30)

    stmt = (
        select(SecurityLog)
        .where(SecurityLog.ts >= desde, SecurityLog.ts <= hasta)
        .order_by(SecurityLog.ts.desc())
        .limit(limite)
    )
    res = await session.execute(stmt)
    logs = list(res.scalars().all())

    if not logs:
        return pd.DataFrame()

    filas = []
    for log in logs:
        usuario = log.usuario or {}
        session_d = log.session or {}
        action = log.action or {}

        ts = log.ts
        hora = ts.hour if ts else 0
        dia = ts.weekday() if ts else 0
        ip = str(session_d.get("ip_address", ""))
        username = str(usuario.get("username", "desconocido"))
        rol = str(usuario.get("role", "desconocido"))
        resource = str(action.get("resource_type", ""))
        componente = RESOURCE_A_COMPONENTE.get(resource, "dashboard")
        operation = str(action.get("operation", ""))
        event_type = str(log.event_type or "")

        filas.append({
            "usuario": username,
            "rol": rol,
            "ip_origen": ip,
            "hora": hora,
            "dia_semana": dia,
            "componente": componente,
            "operation": operation,
            "event_type": event_type,
            "severity": log.severity,
            "es_horario_laboral": _hora_laboral(hora),
            "ip_externa": _ip_externa(ip),
            "ts": ts,
        })

    df_raw = pd.DataFrame(filas)

    # Agregacion por (usuario, hora, componente)
    agg = (
        df_raw.groupby(["usuario", "rol", "componente", "hora", "dia_semana"], as_index=False)
        .agg(
            num_peticiones_hora=("usuario", "size"),
            intentos_fallidos=("event_type", lambda s: int((s == "login_fail").sum())),
            errores_5xx=("severity", lambda s: int((s == "CRITICAL").sum())),
            errores_4xx=("severity", lambda s: int((s == "WARNING").sum())),
            accesos_rutas_sensibles=("componente", lambda s: int((s.isin(["api_auth", "api_modelos"])).sum())),
            usuarios_distintos_ip=("usuario", "nunique"),
            duracion_sesion_min=("ts", lambda s: (s.max() - s.min()).total_seconds() / 60 if len(s) > 1 else 0.0),
            bytes_transferidos_mb=("usuario", lambda s: 0.0),  # no disponible en logs
        )
    )

    # IP representativa por usuario (la mas frecuente)
    ip_por_usuario = (
        df_raw.groupby("usuario")["ip_origen"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
        .to_dict()
    )
    agg["ip_origen"] = agg["usuario"].map(ip_por_usuario)
    agg["es_horario_laboral"] = agg["hora"].apply(_hora_laboral)
    agg["protocolo"] = "HTTPS"  # no disponible en logs
    agg["is_anomalous"] = 0  # los logs reales no traen etiqueta

    # Rellenar columnas que el dataset sintetico tiene y el real no
    for col in ["intentos_fallidos", "errores_4xx", "errores_5xx", "accesos_rutas_sensibles"]:
        if col not in agg.columns:
            agg[col] = 0

    # Orden de columnas equivalente al sintetico
    columnas = [
        "usuario", "rol", "ip_origen", "hora", "dia_semana", "componente",
        "duracion_sesion_min", "num_peticiones_hora", "intentos_fallidos",
        "errores_4xx", "errores_5xx", "bytes_transferidos_mb",
        "accesos_rutas_sensibles", "usuarios_distintos_ip",
        "es_horario_laboral", "protocolo", "is_anomalous",
    ]
    for col in columnas:
        if col not in agg.columns:
            agg[col] = 0
    return agg[columnas]


async def guardar_dataset_real(
    session: AsyncSession,
    df: pd.DataFrame,
    *,
    ruta=config.RESULTS_DIR / "dataset_real.csv",
) -> str:
    """Guarda el DataFrame extraido y devuelve el hash SHA256."""
    df.to_csv(ruta, index=False)
    contenido = ruta.read_bytes()
    return hashlib.sha256(contenido).hexdigest()