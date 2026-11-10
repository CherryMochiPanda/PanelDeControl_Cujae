import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='{"service":"api","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Arranca schedulers (recalculo diario + NBM semanal) y los detiene al apagar.

    Args:
        _app: instancia FastAPI.
    """
    from app.services import scheduler
    from app.services.scheduler import programar, programar_nbm

    # 1. Programar recalculo de indicadores leyendo la config de la BD.
    #    Si falla (BD no disponible), cae al horario por defecto 02:00.
    try:
        from app.database import SessionLocal
        from app.services.scheduler import programar_desde_config

        async with SessionLocal() as session:
            await programar_desde_config(session)
    except Exception as exc:  # noqa: BLE001 - BD no disponible al arrancar
        logging.getLogger("main").warning("No se pudo leer config inicial: %s", exc)
        programar(hora="02:00")

    # 2. Programar reentrenamiento semanal del NBM (independiente del paso 1).
    programar_nbm()

    scheduler.iniciar()
    yield
    scheduler.detener(wait=False)


def create_app() -> FastAPI:
    """Fabrica la aplicacion FastAPI.

    Returns:
        FastAPI: instancia con routers v1 montados en /api/v1.
    """
    app = FastAPI(
        title="CUJAE Cuadro de Mando Institucional",
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()