"""Scheduler de recalculo (RF-36 a RF-40).

Programa recalcular_todos() con aislamiento de fallos. Lee la configuracion
desde config.configuracion_sistema:

- 'intervalo_recalculo_horas' (entero 1..168): ejecuta cada N horas.
- 'hora_recalculo' (HH:MM): modo diario (fallback si no hay intervalo).
"""
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import SessionLocal
from app.models.catalogos import ConfiguracionSistema
from app.services.calculation_engine import recalcular_todos

log = logging.getLogger("scheduler")

scheduler = AsyncIOScheduler(timezone="America/Havana")
_ESTADO_ACTUAL: str | None = None

CLAVE_INTERVALO = "intervalo_recalculo_horas"
CLAVE_HORA = "hora_recalculo"
CLAVE_NBM_HORA = "nbm_reentrenamiento_hora"  # opcional, "03:00" por defecto

async def job_recalculo() -> None:
    """Ejecuta el recalc programado; loguea resultado y errores."""
    async with SessionLocal() as session:
        resultado = await recalcular_todos(session)
    log.info("Recalculo programado: %s", resultado)


def parsear_intervalo(valor: str | None) -> int | None:
    """Convierte el valor de config a horas; None si no es valido.

    Args:
        valor: texto crudo de la config (p. ej. '6').

    Returns:
        Horas del intervalo o None.
    """
    if valor and valor.strip().isdigit():
        horas = int(valor.strip())
        if 1 <= horas <= 168:
            return horas
    return None


def describir(intervalo_horas: int | None, hora: str | None) -> str:
    """Descripcion legible de la frecuencia programada.

    Args:
        intervalo_horas: horas del intervalo o None (modo diario).
        hora: 'HH:MM' del modo diario.

    Returns:
        Texto para la vista de calculo.
    """
    if intervalo_horas:
        return f"Cada {intervalo_horas} horas"
    hh, mm = (hora or "02:00").split(":")
    return f"Diario a las {int(hh):02d}:{int(mm):02d}"


def programar(*, intervalo_horas: int | None = None, hora: str | None = None) -> None:
    """(Re)programa el job de recalculo.

    Args:
        intervalo_horas: si se indica (1..168), usa IntervalTrigger (prioritario).
        hora: 'HH:MM' en zona horaria institucional para modo diario.
    """
    global _ESTADO_ACTUAL
    if intervalo_horas:
        estado = f"cada {intervalo_horas}h"
        trigger: CronTrigger | IntervalTrigger = IntervalTrigger(hours=intervalo_horas)
    else:
        hh, mm = (hora or "02:00").split(":")
        estado = f"diario {int(hh):02d}:{int(mm):02d}"
        trigger = CronTrigger(hour=int(hh), minute=int(mm))
    if _ESTADO_ACTUAL == estado:
        return
    scheduler.add_job(job_recalculo, trigger, id="recalculo_diario", replace_existing=True)
    _ESTADO_ACTUAL = estado
    log.info("Recalculo programado: %s", estado)




async def job_reentrenar_nbm() -> None:
    """Job semanal: reentrena el modelo OC-SVM del NBM."""
    from app.nbm import training
    try:
        metricas = await training.entrenar_y_guardar_oneclasssvm(fuente="sintetico")
        log.info("NBM reentrenado: %s", metricas)
    except Exception as exc:  # noqa: BLE001
        log.exception("Reentrenamiento NBM fallo: %s", exc)


def programar_nbm(*, dia_semana: str = "mon", hora: str = "03:00") -> None:
    """Programa el reentrenamiento semanal del NBM (default: lunes 03:00).

    Args:
        dia_semana: 'mon', 'tue', ..., 'sun'.
        hora: 'HH:MM' en zona horaria institucional.
    """
    hh, mm = hora.split(":")
    trigger = CronTrigger(day_of_week=dia_semana, hour=int(hh), minute=int(mm))
    scheduler.add_job(
        job_reentrenar_nbm, trigger,
        id="nbm_retrain_weekly", replace_existing=True,
    )
    log.info("NBM reentrenamiento programado: %s %s", dia_semana, hora)

async def programar_desde_config(session: AsyncSession) -> None:
    """(Re)programa el job leyendo las claves de configuracion de la BD.

    Args:
        session: sesion async con acceso a config.configuracion_sistema.
    """
    res = await session.execute(
        select(ConfiguracionSistema.clave, ConfiguracionSistema.valor).where(
            ConfiguracionSistema.clave.in_((CLAVE_INTERVALO, CLAVE_HORA))
        )
    )
    valores = dict(res.all())
    programar(
        intervalo_horas=parsear_intervalo(valores.get(CLAVE_INTERVALO)),
        hora=valores.get(CLAVE_HORA, "02:00"),
    )


def proxima_ejecucion() -> str | None:
    """ISO8601 de la proxima ejecucion del job programado.

    Returns:
        Fecha/hora o None si el job no esta encolado.
    """
    job = scheduler.get_job("recalculo_diario")
    if job is not None and job.next_run_time is not None:
        return job.next_run_time.isoformat()
    return None


def iniciar() -> None:
    """Arranca el scheduler (llamado desde lifespan de FastAPI)."""
    if not scheduler.running:
        scheduler.start()


detener = scheduler.shutdown
