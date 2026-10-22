"""Extraccion de features desde logs reales (auditoria.security_logs).

Lee los logs JSONB de PostgreSQL y construye un DataFrame con el mismo
esquema que el dataset sintetico (data_generator.py), para que el runner
pueda entrenar con datos reales sin cambiar los algoritmos.

Los logs ya vienen seudonimizados por write_security_log:
- usuario.username -> usuario.username_hash (HMAC-SHA256 con pepper)
- session.ip_address -> session.ip_hash (HMAC-SHA256 con pepper)
- session.ip_externa -> flag binaria (calculada antes de anonimizar)
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.auditoria import SecurityLog
from app.nbm import config

RESOURCE_A_COMPONENTE = {
    "usuario": "api_auth",
    "configuracion": "api_modelos",
    "indicador": "api_indicadores",
    "fuente": "api_export",
    "proceso": "api_indicadores",
    "curso": "api_indicadores",
}


def _hora_laboral(hora: int) -> int:
    return 1 if 8 <= hora < 19 else 0


async def extraer_features_reales(
    session: AsyncSession,
    *,
    desde: datetime | None = None,
    hasta: datetime | None = None,
    limite: int = 50000,
) -> pd.DataFrame:
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
        sess = log.session or {}
        action = log.action or {}
        ts = log.ts
        hora = ts.hour if ts else 0
        dia = ts.weekday() if ts else 0

        # Campos seudonimizados por write_security_log
        username_hash = str(usuario.get("username_hash", "desconocido"))
        rol = str(usuario.get("role", "desconocido"))
        ip_externa = int(sess.get("ip_externa", 0))

        resource = str(action.get("resource_type", ""))
        componente = RESOURCE_A_COMPONENTE.get(resource, "dashboard")

        filas.append({
            "usuario": username_hash,
            "rol": rol,
            "ip_origen": "",  # reemplazado por ip_externa
            "hora": hora,
            "dia_semana": dia,
            "componente": componente,
            "ip_externa": ip_externa,
            "event_type": str(log.event_type or ""),
            "severity": log.severity,
            "es_horario_laboral": _hora_laboral(hora),
            "ts": ts,
        })

    df_raw = pd.DataFrame(filas)

    agg = (
        df_raw.groupby(["usuario", "rol", "componente", "hora", "dia_semana"], as_index=False)
        .agg(
            num_peticiones_hora=("usuario", "size"),
            intentos_fallidos=("event_type", lambda s: int((s == "login_fail").sum())),
            errores_5xx=("severity", lambda s: int((s == "CRITICAL").sum())),
            errores_4xx=("severity", lambda s: int((s == "WARNING").sum())),
            accesos_rutas_sensibles=("componente", lambda s: int((s.isin(["api_auth", "api_modelos"])).sum())),
            usuarios_distintos_ip=("usuario", "nunique"),
            ip_externa=("ip_externa", "max"),
            duracion_sesion_min=("ts", lambda s: (s.max() - s.min()).total_seconds() / 60 if len(s) > 1 else 0.0),
            bytes_transferidos_mb=("usuario", lambda s: 0.0),
        )
    )
    agg["es_horario_laboral"] = agg["hora"].apply(_hora_laboral)
    agg["protocolo"] = "HTTPS"
    agg["is_anomalous"] = 0

    columnas = [
        "usuario", "rol", "ip_origen", "hora", "dia_semana", "componente",
        "duracion_sesion_min", "num_peticiones_hora", "intentos_fallidos",
        "errores_4xx", "errores_5xx", "bytes_transferidos_mb",
        "accesos_rutas_sensibles", "usuarios_distintos_ip",
        "es_horario_laboral", "protocolo", "ip_externa", "is_anomalous",
    ]
    for col in columnas:
        if col not in agg.columns:
            agg[col] = 0
    return agg[columnas]


async def guardar_dataset_real(
    session,
    df: pd.DataFrame,
    *,
    ruta=config.RESULTS_DIR / "dataset_real.csv",
) -> str:
    df.to_csv(ruta, index=False)
    return hashlib.sha256(ruta.read_bytes()).hexdigest()