"""Script CLI que orquesta el experimento completo del NBM.

Uso:
    python -m app.nbm.runner
    python -m app.nbm.runner --regenerate
    python -m app.nbm.runner --skip-autoencoder
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.database import SessionLocal
from app.nbm import algorithms, config, data_generator, evaluator, persistence


def _preparar_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convierte el DataFrame a matriz numerica X y vector y.

    - One-hot encoding de columnas categoricas (usuario, rol, componente, protocolo).
    - ip_origen se transforma en feature binaria 'ip_externa' (no empieza por 192.168.).
    - Se eliminan columnas no numericas tras el encoding.
    """
    df = df.copy()
    df["ip_externa"] = (~df["ip_origen"].astype(str).str.startswith("192.168.")).astype(int)
    df = df.drop(columns=["ip_origen"])
    y = df["is_anomalous"].astype(int).to_numpy()
    X_df = df.drop(columns=["is_anomalous"])
    X_df = pd.get_dummies(
        X_df, columns=["usuario", "rol", "componente", "protocolo"], drop_first=False,
    )
    features = X_df.columns.tolist()
    return X_df.to_numpy(dtype=np.float64), y, features


async def _persistir(
    metricas: list[evaluator.Metricas],
    *,
    dataset_id,
    n_train: int,
    n_test: int,
) -> None:
    async with SessionLocal() as session:
        for m in metricas:
            await persistence.registrar_experimento(
                session, m, dataset_id=dataset_id, n_train=n_train, n_test=n_test,
            )
        await session.commit()


def _imprimir_resumen(df: pd.DataFrame, mejor: str) -> None:
    print("\n" + "=" * 100)
    print("TABLA GENERAL - COMPARATIVA DE ALGORITMOS - NBM CUJAE")
    print("=" * 100)
    cols = ["algoritmo", "precision", "recall", "f1", "fpr",
            "tiempo_entrenamiento_s", "tiempo_prediccion_s"]
    print(df[cols].to_string(index=False))
    print("-" * 100)
    print(f"MEJOR ALGORITMO (F1): {mejor}")
    print("=" * 100 + "\n")


def _imprimir_detalle(metricas: list[evaluator.Metricas]) -> None:
    """Imprime una tabla detallada por cada algoritmo (formato vertical)."""
    for m in metricas:
        print("\n" + "#" * 100)
        print(f"# DETALLE POR ALGORITMO: {m.algoritmo}")
        print("#" * 100)
        tabla = evaluator.tabla_detallada(m)
        print(tabla.to_string(index=False))
        if m.extra:
            print("\nExtra:")
            for k, v in m.extra.items():
                print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

async def main(regenerar: bool, skip_autoencoder: bool) -> int:
    # 1. Dataset
    if regenerar or not config.DATASET_CSV.exists():
        print(f"[1/5] Generando dataset sintetico ({config.N_REGISTROS} registros)...")
        df = data_generator.generar_dataset_sintetico()
        h = data_generator.guardar_dataset(df)
        fuente = "sintetico"
        seed = config.SEED
        descripcion = "Generado por app.nbm.data_generator"
    else:
        print(f"[1/5] Reutilizando dataset existente: {config.DATASET_CSV}")
        df = pd.read_csv(config.DATASET_CSV)
        import hashlib
        h = hashlib.sha256(config.DATASET_CSV.read_bytes()).hexdigest()
        fuente = "sintetico"
        seed = config.SEED
        descripcion = "Cargado desde CSV"

    print(f"      Registros: {len(df)} | Anomalias: {int(df['is_anomalous'].sum())}")

    # 2. Features
    print("[2/5] Preparando features (encoding + escalado)...")
    X, y, features = _preparar_xy(df)
    print(f"      Features tras encoding: {len(features)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Algoritmos
    print("[3/5] Entrenando algoritmos...")
    resultados = []
    errores = []

    for nombre, fn in [
        ("IsolationForest", algorithms.ejecutar_isolation_forest),
        ("OneClassSVM", algorithms.ejecutar_one_class_svm),
    ]:
        try:
            print(f"      - {nombre}...", end=" ", flush=True)
            r = fn(X_train_s, y_train, X_test_s)
            resultados.append(r)
            print("ok")
        except Exception as exc:
            print(f"ERROR: {exc}")
            errores.append(f"{nombre}: {exc}")

    if not skip_autoencoder:
        try:
            print("      - Autoencoder...", end=" ", flush=True)
            r = algorithms.ejecutar_autoencoder(X_train_s, y_train, X_test_s)
            resultados.append(r)
            print("ok")
        except Exception as exc:
            print(f"ERROR: {exc}")
            errores.append(f"Autoencoder: {exc}")

    if not resultados:
        print("\n[FATAL] Ningun algoritmo termino. Revisa los errores:")
        for e in errores:
            print(f"  - {e}")
        return 1

    # 4. Evaluacion
    print("[4/5] Evaluando y comparando...")
    metricas = [evaluator.evaluar(r, y_test) for r in resultados]
    df_cmp = evaluator.comparar(metricas)
    mejor = evaluator.seleccionar_mejor(df_cmp)

    # 5. Persistencia
    print("[5/5] Persistiendo en PostgreSQL (schema nbm)...")
    async with SessionLocal() as session:
        dataset_id = await persistence.registrar_dataset(
            session,
            fuente=fuente,
            n_registros=len(df),
            n_anomalias=int(df["is_anomalous"].sum()),
            n_features=len(features),
            hash_sha256=h,
            ruta_csv=str(config.DATASET_CSV),
            seed=seed,
            descripcion=descripcion,
        )
        await session.commit()

    await _persistir(metricas, dataset_id=dataset_id, n_train=len(X_train), n_test=len(X_test))

    df_cmp.to_csv(config.COMPARATIVA_CSV, index=False)
    print(f"      Resultados en {config.COMPARATIVA_CSV}")
    print(f"      Registrados en nbm.experimentos (dataset_id={dataset_id})")

    _imprimir_resumen(df_cmp, mejor)
    _imprimir_detalle(metricas)
    if errores:
        print("Advertencias:")
        for e in errores:
            print(f"  - {e}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimento NBM: comparar algoritmos.")
    parser.add_argument("--regenerate", action="store_true", help="Fuerza regenerar dataset sintetico.")
    parser.add_argument("--skip-autoencoder", action="store_true", help="Omite el Autoencoder.")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.regenerate, args.skip_autoencoder)))