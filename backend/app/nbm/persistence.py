"""Persistencia de resultados en PostgreSQL (schema nbm)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.nbm import DatasetGenerado, Experimento
from app.nbm.evaluator import Metricas


async def registrar_dataset(
    session: AsyncSession,
    *,
    fuente: str,
    n_registros: int,
    n_anomalias: int,
    n_features: int,
    hash_sha256: str,
    ruta_csv: str | None,
    seed: int | None,
    descripcion: str | None = None,
) -> uuid.UUID:
    """Inserta o reutiliza un registro de dataset (por hash)."""
    res = await session.execute(
        select(DatasetGenerado).where(DatasetGenerado.hash_sha256 == hash_sha256)
    )
    existente = res.scalar_one_or_none()
    if existente is not None:
        return existente.id
    ds = DatasetGenerado(
        id=uuid.uuid4(), fuente=fuente,
        n_registros=n_registros, n_anomalias=n_anomalias, n_features=n_features,
        hash_sha256=hash_sha256, ruta_csv=ruta_csv, seed=seed, descripcion=descripcion,
    )
    session.add(ds)
    await session.flush()
    return ds.id


async def registrar_experimento(
    session: AsyncSession,
    metrica: Metricas,
    *,
    dataset_id: uuid.UUID,
    n_train: int,
    n_test: int,
    notas: str | None = None,
) -> uuid.UUID:
    """Inserta una fila de experimento con sus metricas."""
    exp = Experimento(
        id=uuid.uuid4(),
        fecha=datetime.now(timezone.utc),
        algoritmo=metrica.algoritmo,
        hiperparametros=metrica.hiperparametros,
        precision=round(metrica.precision, 4),
        recall=round(metrica.recall, 4),
        f1_score=round(metrica.f1, 4),
        fpr=round(metrica.fpr, 4),
        auc_roc=round(metrica.auc_roc, 4) if metrica.auc_roc is not None else None,
        tp=metrica.tp, tn=metrica.tn, fp=metrica.fp, fn=metrica.fn,
        tiempo_entrenamiento_s=round(metrica.tiempo_entrenamiento_s, 4),
        tiempo_prediccion_s=round(metrica.tiempo_prediccion_s, 4),
        n_train=n_train, n_test=n_test,
        dataset_id=dataset_id, notas=notas,
    )
    session.add(exp)
    await session.flush()
    return exp.id