import socket
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.auditoria import Auditoria, SecurityLog
from app.security.anonymizer import anonymize

VERSION = "0.1.0"


async def write_security_log(
    session: AsyncSession,
    *,
    event_type: str,
    severity: str = "INFO",
    user: dict[str, Any] | None = None,
    ip: str = "",
    user_agent: str = "",
    resource_type: str = "",
    resource_id: str = "",
    operation: str = "",
    details: dict[str, Any] | None = None,
    commit: bool = True,
) -> None:
    """Escribe un log estructurado JSON con datos seudonimizados.

    Aplica HMAC-SHA256 con pepper (NBM_ANONYMIZATION_PEPPER) a username e IP
    antes de persistir en auditoria.security_logs. Preserva la correlacion
    por usuario sin almacenar identificadores directos (ver Informe PP2 1.3.5).

    Args:
        session: sesion async.
        event_type: tipo de evento (login, indicator_create, ...).
        severity: INFO|WARNING|CRITICAL.
        user: payload de usuario (id, username, role, auth_type, email).
        ip: direccion IP del cliente.
        user_agent: cabecera User-Agent.
        resource_type: recurso afectado.
        resource_id: id del recurso.
        operation: create|read|update|delete|login|logout.
        details: detalles adicionales.
        commit: si True hace commit inmediato.
    """
    # Seudonimizar identificadores sensibles antes de persistir
    user_seguro: dict[str, Any] = {}
    if user:
        user_seguro = {
            "user_id": str(user.get("user_id", "")),  # UUID ya es seudonimo interno
            "username_hash": anonymize(user.get("username", "")),
            "role": user.get("role", ""),
            "auth_type": user.get("auth_type", ""),
            # email no se persiste en logs (minimizacion)
        }

    log = SecurityLog(
        log_id=uuid.uuid4(),
        ts=datetime.now(timezone.utc),
        event_type=event_type,
        severity=severity,
        usuario=user_seguro,
        session={
            "session_id": str(uuid.uuid4()),
            "ip_hash": anonymize(ip) if ip else "",
            # Bandera calculada ANTES de hashear (necesaria para features del NBM)
            "ip_externa": 0 if (ip or "").startswith("192.168.") else 1,
            "user_agent": user_agent,
        },
        action={
            "resource_type": resource_type,
            "resource_id": resource_id,
            "operation": operation,
            "details": details or {},
        },
        system={
            "hostname": socket.gethostname(),
            "service": "api",
            "version": VERSION,
        },
    )
    session.add(log)
    if commit:
        await session.commit()


async def write_audit(
    session: AsyncSession,
    *,
    usuario_id: uuid.UUID | None,
    accion: str,
    recurso_tipo: str,
    recurso_id: uuid.UUID | None = None,
    detalles: dict[str, Any] | None = None,
    commit: bool = True,
) -> None:
    """Escribe en la tabla append-only de auditoria (RF-51/52).

    Args:
        session: sesion async.
        usuario_id: usuario responsable.
        accion: CREATE/UPDATE/DELETE/LOGIN_*/RECALC...
        recurso_tipo: tipo de recurso.
        recurso_id: id del recurso.
        detalles: payload JSON.
        commit: si True hace commit inmediato.
    """
    session.add(
        Auditoria(
            usuario_id=usuario_id,
            accion=accion,
            recurso_tipo=recurso_tipo,
            recurso_id=recurso_id,
            detalles=detalles,
        )
    )
    if commit:
        await session.commit()