"""Entrenamiento y guardado del modelo OneClassSVM para produccion.

Reutiliza el pipeline del runner (dataset, preparacion de features) pero
solo entrena OC-SVM (el ganador de la comparacion) y guarda el artefacto.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.database import SessionLocal
from app.nbm import algorithms, config, data_generator, evaluator, model_store, persistence

log = logging.getLogger("nbm.training")


async def entrenar_y_guardar_oneclasssvm(*, fuente: str = "sintetico") -> dict:
    """Entrena OC-SVM sobre el dataset actual y guarda el modelo.

    Args:
        fuente: "sintetico" (default) o "real".

    Returns:
        Dict con metricas del modelo entrenado.
    """
    # 1. Cargar dataset
    if fuente == "real":
        from app.nbm import extractors, anomaly_injector
        async with SessionLocal() as session:
            df = await extractors.extraer_features_reales(session)
        if df.empty:
            raise RuntimeError("No hay logs reales para entrenar.")
        df = anomaly_injector.inyectar_anomalias(df)
    else:
        if not config.DATASET_CSV.exists():
            df = data_generator.generar_dataset_sintetico()
            data_generator.guardar_dataset(df)
        else:
            df = pd.read_csv(config.DATASET_CSV)

    # 2. Preparar features (mismo codigo que el runner)
    from app.nbm.runner import _preparar_xy
    X, y, features = _preparar_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 3. Entrenar OC-SVM
    resultado = await asyncio_to_thread(
        algorithms.ejecutar_one_class_svm, X_train_s, y_train, X_test_s,
    )
    metricas = evaluator.evaluar(resultado, y_test)

    # 4. Guardar modelo
    metadata = {
        "algoritmo": "OneClassSVM",
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(features),
        "precision": metricas.precision,
        "recall": metricas.recall,
        "f1": metricas.f1,
        "fpr": metricas.fpr,
        "hiperparametros": resultado.hiperparametros,
        "fuente": fuente,
    }
    model_store.save_model(
        model=resultado.extra["model"],  # ver nota abajo
        scaler=scaler,
        features=features,
        metadata=metadata,
    )

    # 5. Persistir experimento en BD
    try:
        async with SessionLocal() as session:
            ds_id = await persistence.registrar_dataset(
                session,
                fuente=f"retrain_{fuente}",
                n_registros=len(df),
                n_anomalias=int(df["is_anomalous"].sum()),
                n_features=len(features),
                hash_sha256=f"retrain_{datetime.now(timezone.utc).timestamp()}",
                ruta_csv=str(config.DATASET_CSV),
                seed=config.SEED,
                descripcion=f"Reentrenamiento {fuente}",
            )
            await persistence.registrar_experimento(
                session, metricas, dataset_id=ds_id,
                n_train=len(X_train), n_test=len(X_test),
                notas="Reentrenamiento automatico",
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo persistir experimento de reentrenamiento: %s", exc)

    log.info("Reentrenamiento completado: F1=%.4f", metricas.f1)
    return {
        "algoritmo": "OneClassSVM",
        "precision": metricas.precision,
        "recall": metricas.recall,
        "f1": metricas.f1,
        "fpr": metricas.fpr,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


async def asyncio_to_thread(fn, *args):
    """Helper para correr funciones sync en un hilo."""
    import asyncio
    return await asyncio.to_thread(fn, *args)