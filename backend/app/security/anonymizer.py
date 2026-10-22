"""Mecanismo de anonimizacion HMAC-SHA256 con pepper.

El pepper se lee de la variable de entorno NBM_ANONYMIZATION_PEPPER y
NUNCA se guarda en la base de datos.
"""
from __future__ import annotations

import hashlib
import hmac

from app.config import settings


def anonymize(value: str | None) -> str:
    """Calcula el HMAC-SHA256 de un valor usando el pepper del sistema.

    Args:
        value: cadena a anonimizar (username, email, IP, etc.).

    Returns:
        Hash hexadecimal de 64 caracteres. Si value es None o vacio,
        devuelve cadena vacia.
    """
    if not value:
        return ""
    pepper = settings.NBM_ANONYMIZATION_PEPPER.encode("utf-8")
    mensaje = str(value).strip().lower().encode("utf-8")
    return hmac.new(pepper, mensaje, hashlib.sha256).hexdigest()