from fastapi import APIRouter

from app.api.v1 import (
    auditoria,
    auth,
    config,
    fuentes,
    indicadores,
    logs,
    procesos,
    usuarios,
)
from app.api.v1.health import router as health_router
from app.api.v1 import nbm 

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(auth.router)
api_router.include_router(indicadores.router)
api_router.include_router(procesos.router)
api_router.include_router(usuarios.router)
api_router.include_router(fuentes.router)
api_router.include_router(auditoria.router)
api_router.include_router(logs.router)
api_router.include_router(config.router)
api_router.include_router(nbm.router) 
