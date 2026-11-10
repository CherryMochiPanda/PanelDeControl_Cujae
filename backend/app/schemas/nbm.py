"""Schemas Pydantic para el endpoint NBM."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

class EventoIn(BaseModel):
    """Evento crudo a evaluar por el NBM."""

    hora: int = Field(ge=0, le=23)
    dia_semana: int = Field(ge=0, le=6)
    es_horario_laboral: int = Field(ge=0, le=1)
    duracion_sesion_min: float = Field(ge=0)
    num_peticiones_hora: int = Field(ge=0)
    intentos_fallidos: int = Field(ge=0)
    errores_4xx: int = Field(ge=0)
    errores_5xx: int = Field(ge=0)
    bytes_transferidos_mb: float = Field(ge=0)
    accesos_rutas_sensibles: int = Field(ge=0)
    usuarios_distintos_ip: int = Field(ge=1)
    ip_externa: int = Field(ge=0, le=1)
    rol: Literal["estudiante", "profesor", "jefe_proceso", "directivo", "administrador", "auditor"]
    componente: Literal["dashboard", "api_auth", "api_indicadores", "api_export", "api_modelos"]
    protocolo: Literal["HTTPS", "SQL", "LDAP"]


class DeteccionOut(BaseModel):
    index: int
    is_anomaly: bool
    score: float


class DetectarIn(BaseModel):
    eventos: list[EventoIn] = Field(min_length=1, max_length=1000)


class DetectarOut(BaseModel):
    total: int
    anomalias: int
    resultados: list[DeteccionOut]


class ModeloInfoOut(BaseModel):
    disponible: bool
    metadata: dict | None = None


class ReentrenarOut(BaseModel):
    status: str
    mensaje: str



class AlertaOut(BaseModel):
    model_config = {"from_attributes": True}

    id: UUID
    fecha: datetime
    usuario_hash: str
    componente: str
    score: float
    severidad: str
    tipo_anomalia: str
    estado: str
    detalles: dict


class ReporteParams(BaseModel):
    desde: datetime | None = None
    hasta: datetime | None = None