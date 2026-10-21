"""Metricas de evaluacion y comparacion de algoritmos."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from app.nbm.algorithms import ResultadoAlgoritmo


@dataclass
class Metricas:
    algoritmo: str
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    fpr: float
    auc_roc: float | None
    tiempo_entrenamiento_s: float
    tiempo_prediccion_s: float
    hiperparametros: dict
    total_analizados: int = 0
    total_anomalias_reales: int = 0
    total_anomalias_capturadas: int = 0
    total_correctamente_clasificados: int = 0
    cpu_entrenamiento_pct: float = 0.0
    cpu_prediccion_pct: float = 0.0
    ram_entrenamiento_mb: float = 0.0
    ram_prediccion_mb: float = 0.0
    extra: dict = field(default_factory=dict)


def evaluar(resultado: ResultadoAlgoritmo, y_test: np.ndarray) -> Metricas:
    """Calcula matriz de confusion y metricas derivadas."""
    tn, fp, fn, tp = confusion_matrix(y_test, resultado.y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_test, resultado.y_pred, zero_division=0)
    recall = recall_score(y_test, resultado.y_pred, zero_division=0)
    f1 = f1_score(y_test, resultado.y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    try:
        auc = roc_auc_score(y_test, resultado.y_pred)
    except ValueError:
        auc = None

    total = len(y_test)
    anomalias_reales = int(np.sum(y_test == 1))
    correctos = int(tp + tn)

    return Metricas(
        algoritmo=resultado.nombre,
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
        precision=float(precision), recall=float(recall), f1=float(f1),
        fpr=float(fpr), auc_roc=auc,
        tiempo_entrenamiento_s=resultado.tiempo_entrenamiento_s,
        tiempo_prediccion_s=resultado.tiempo_prediccion_s,
        hiperparametros=resultado.hiperparametros,
        total_analizados=total,
        total_anomalias_reales=anomalias_reales,
        total_anomalias_capturadas=int(tp),
        total_correctamente_clasificados=correctos,
        cpu_entrenamiento_pct=resultado.cpu_entrenamiento_pct,
        cpu_prediccion_pct=resultado.cpu_prediccion_pct,
        ram_entrenamiento_mb=resultado.ram_entrenamiento_mb,
        ram_prediccion_mb=resultado.ram_prediccion_mb,
        extra=resultado.extra,
    )


def comparar(metricas: list[Metricas]) -> pd.DataFrame:
    """Devuelve DataFrame comparativo ordenado por F1 desc (tabla general)."""
    columnas = [
        "algoritmo", "precision", "recall", "f1", "fpr",
        "tiempo_entrenamiento_s", "tiempo_prediccion_s",
    ]
    df = pd.DataFrame([m.__dict__ for m in metricas])
    df = df[columnas].sort_values("f1", ascending=False).reset_index(drop=True)
    return df


def tabla_detallada(m: Metricas) -> pd.DataFrame:
    """Tabla detallada de un solo algoritmo con todos los campos solicitados."""
    filas = [
        ("Algoritmo", m.algoritmo),
        ("Total de datos analizados (test)", m.total_analizados),
        ("Total de anomalías reales", m.total_anomalias_reales),
        ("Total de anomalías capturadas (TP)", m.total_anomalias_capturadas),
        ("Total correctamente clasificados (TP+TN)", m.total_correctamente_clasificados),
        ("Falsos positivos (FP)", m.fp),
        ("Falsos negativos (FN)", m.fn),
        ("Precisión", f"{m.precision:.4f}"),
        ("Recall", f"{m.recall:.4f}"),
        ("F1-Score", f"{m.f1:.4f}"),
        ("Tasa de falsos positivos (FPR)", f"{m.fpr:.4f}"),
        ("Tiempo de entrenamiento (s)", f"{m.tiempo_entrenamiento_s:.4f}"),
        ("Tiempo de predicción (s)", f"{m.tiempo_prediccion_s:.4f}"),
        ("CPU entrenamiento (% máquina)", f"{m.cpu_entrenamiento_pct:.2f}"),
        ("CPU predicción (% máquina)", f"{m.cpu_prediccion_pct:.2f}"),
        ("RAM entrenamiento (MB incremento)", f"{m.ram_entrenamiento_mb:.2f}"),
        ("RAM predicción (MB incremento)", f"{m.ram_prediccion_mb:.2f}"),
    ]
    return pd.DataFrame(filas, columns=["Métrica", "Valor"])


def seleccionar_mejor(df: pd.DataFrame) -> str:
    """Selecciona el algoritmo con mejor F1; desempata por menor FPR."""
    df_ord = df.sort_values(["f1", "fpr"], ascending=[False, True])
    return str(df_ord.iloc[0]["algoritmo"])