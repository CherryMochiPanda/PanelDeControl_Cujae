"""Endpoints del NBM: deteccion, reentrenamiento, info del modelo."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import MeOut, require
from app.nbm import detection_service, model_store
from app.schemas.nbm import (
    DetectarIn, DetectarOut, ModeloInfoOut, ReentrenarOut,
)
import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.nbm import Alerta
from fastapi.responses import StreamingResponse
from datetime import datetime
from app.services import report_service

router = APIRouter(prefix="/nbm", tags=["nbm"])
log = logging.getLogger("nbm.api")





@router.post("/detectar", response_model=DetectarOut)
async def detectar(
    body: DetectarIn,
    _: MeOut = Depends(require("logs")),
    session: AsyncSession = Depends(get_db),
) -> DetectarOut:
    """Detecta anomalias en un lote y persiste las alertas generadas."""
    eventos = [e.model_dump() for e in body.eventos]
    try:
        resultados = detection_service.detectar_eventos(eventos)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    anomalias = 0
    for r in resultados:
        if not r["is_anomaly"]:
            continue
        anomalias += 1
        ev = eventos[r["index"]]
        score = r["score"]

        # Inferir severidad a partir del score (más negativo = más grave)
        if score < -3:
            severidad = "CRITICAL"
        elif score < -1.5:
            severidad = "HIGH"
        elif score < -0.5:
            severidad = "MEDIUM"
        else:
            severidad = "LOW"

        # Inferir tipo de anomalía por el componente y features
        if ev.get("componente") == "api_auth" and ev.get("intentos_fallidos", 0) > 5:
            tipo = "fuerza_bruta"
        elif ev.get("errores_4xx", 0) > 20:
            tipo = "escaneo"
        elif ev.get("bytes_transferidos_mb", 0) > 50:
            tipo = "exfiltracion"
        elif ev.get("hora", 12) < 6 or ev.get("hora", 12) > 22:
            tipo = "acceso_nocturno"
        elif ev.get("usuarios_distintos_ip", 1) > 5:
            tipo = "password_spraying"
        else:
            tipo = "comportamiento_anomalo"

        session.add(Alerta(
            usuario_hash="anon_" + str(uuid.uuid4())[:16],  # placeholder hasta tener hash real
            componente=ev.get("componente", "desconocido"),
            score=score,
            severidad=severidad,
            tipo_anomalia=tipo,
            estado="nueva",
            detalles=ev,
        ))

    await session.commit()

    return DetectarOut(
        total=len(eventos),
        anomalias=anomalias,
        resultados=resultados,
    )


@router.post("/reentrenar", response_model=ReentrenarOut, status_code=status.HTTP_202_ACCEPTED)
async def reentrenar(_: MeOut = Depends(require("config"))) -> ReentrenarOut:
    """Dispara un reentrenamiento del modelo en background.

    Solo accesible para administradores (permiso 'config').
    El reentrenamiento corre en un hilo aparte para no bloquear el request.
    """
    from app.nbm import training  # import local para evitar ciclo

    async def _job():
        try:
            await training.entrenar_y_guardar_oneclasssvm(fuente="sintetico")
        except Exception as exc:  # noqa: BLE001
            log.exception("Reentrenamiento NBM fallo: %s", exc)

    asyncio.create_task(_job())
    return ReentrenarOut(
        status="accepted",
        mensaje="Reentrenamiento iniciado en background. Consulta /nbm/modelo/info en unos minutos.",
    )


@router.get("/modelo/info", response_model=ModeloInfoOut)
def modelo_info(_: MeOut = Depends(require("logs"))) -> ModeloInfoOut:
    """Devuelve metadata del modelo actual (o indica que no hay)."""
    metadata = model_store.get_metadata()
    return ModeloInfoOut(
        disponible=metadata is not None,
        metadata=metadata,
    )




@router.get("/reportes")
async def generar_reporte(
    desde: datetime | None = None,
    hasta: datetime | None = None,
    _: MeOut = Depends(require("logs")),
    session: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Genera un reporte Excel de anomalías en el rango de fechas.

    Args:
        desde: fecha inicio (ISO 8601, default: hace 7 días).
        hasta: fecha fin (ISO 8601, default: ahora).
        _: usuario con permiso de logs.
        session: sesión async.

    Returns:
        Archivo .xlsx para descarga, o 404 si no hay alertas.

    Raises:
        HTTPException 404: si no hay alertas en el rango.
    """
    contenido = await report_service.generar_reporte_anomalias(
        session, desde=desde, hasta=hasta,
    )
    if contenido is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No hay anomalías registradas en el período solicitado.",
        )
    return StreamingResponse(
        iter([contenido]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename=reporte_anomalias_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
        },
    )