"""Consulta y exportacion de security_logs estructurados (JSON)."""
import csv
import io
import json
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import MeOut, require_logs
from app.database import get_db
from app.models.auditoria import SecurityLog
from app.schemas.operaciones import SecurityLogOut

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("/security", response_model=list[SecurityLogOut])
async def listar_security_logs(
    event_type: str | None = None,
    severidad: str | None = None,
    usuario: str | None = None,
    desde: datetime | None = None,
    hasta: datetime | None = None,
    limite: int = Query(default=200, ge=1, le=1000),
    _: MeOut = Depends(require_logs),
    session: AsyncSession = Depends(get_db),
) -> list[SecurityLog]:
    """Logs estructurados JSON con filtros por fecha/usuario/evento/severidad.

    Args:
        event_type: tipo de evento.
        severidad: severidad.
        usuario: username exacto.
        desde: fecha inicio.
        hasta: fecha fin.
        limite: maximo de filas.
        _: usuario con permiso.
        session: sesion async.

    Returns:
        Logs ordenados por tiempo.
    """
    stmt = select(SecurityLog).order_by(SecurityLog.ts.desc()).limit(limite)
    if event_type:
        stmt = stmt.where(SecurityLog.event_type == event_type)
    if severidad:
        stmt = stmt.where(SecurityLog.severity == severidad)
    if usuario:
        stmt = stmt.where(SecurityLog.usuario["username_hash"].as_string() == usuario)
    if desde:
        stmt = stmt.where(SecurityLog.ts >= desde)
    if hasta:
        stmt = stmt.where(SecurityLog.ts <= hasta)
    res = await session.execute(stmt)
    return list(res.scalars().all())  # type: ignore[return-value]


@router.get("/export")
async def exportar_logs(
    formato: str = Query(default="json", pattern="^(json|csv)$"),
    desde: datetime | None = None,
    hasta: datetime | None = None,
    _: MeOut = Depends(require_logs),
    session: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Exporta security_logs a JSON/CSV (limite RC-9: 2MB).

    Args:
        formato: json|csv.
        desde: fecha inicio.
        hasta: fecha fin.
        _: usuario con permiso.
        session: sesion async.

    Returns:
        StreamingResponse con el archivo.
    """
    stmt = select(SecurityLog).order_by(SecurityLog.ts.desc()).limit(5000)
    if desde:
        stmt = stmt.where(SecurityLog.ts >= desde)
    if hasta:
        stmt = stmt.where(SecurityLog.ts <= hasta)
    res = await session.execute(stmt)
    logs = list(res.scalars().all())

    if formato == "json":
        payload = json.dumps(
            [
                {
                    "log_id": str(l.log_id),
                    "ts": l.ts.isoformat(),
                    "event_type": l.event_type,
                    "severity": l.severity,
                    "usuario": l.usuario,
                    "session": l.session,
                    "action": l.action,
                    "system": l.system,
                }
                for l in logs
            ],
            ensure_ascii=False,
            indent=2,
        )
        return StreamingResponse(
            iter([payload]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=security_logs.json"},
        )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["log_id", "ts", "event_type", "severity", "username_hash", "ip", "recurso", "operacion"]
    )
    for l in logs:
        writer.writerow(
            [
                l.log_id,
                l.ts.isoformat(),
                l.event_type,
                l.severity,
                l.usuario.get("username_hash", ""),
                l.session.get("ip_address", ""),
                l.action.get("resource_type", ""),
                l.action.get("operation", ""),
            ]
        )
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=security_logs.csv"},
    )
