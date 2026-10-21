"""Metricas de evaluacion y comparacion de algoritmos."""
from __future__ import annotations

import time
from dataclasses import dataclass

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
    return Metricas(
        algoritmo=resultado.nombre,
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
        precision=float(precision), recall=float(recall), f1=float(f1),
        fpr=float(fpr), auc_roc=auc,
        tiempo_entrenamiento_s=resultado.tiempo_entrenamiento_s,
        tiempo_prediccion_s=resultado.tiempo_prediccion_s,
        hiperparametros=resultado.hiperparametros,
    )


def comparar(metricas: list[Metricas]) -> pd.DataFrame:
    """Devuelve DataFrame comparativo ordenado por F1 desc."""
    df = pd.DataFrame([m.__dict__ for m in metricas])
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    return df


def seleccionar_mejor(df: pd.DataFrame) -> str:
    """Selecciona el algoritmo con mejor F1; desempata por menor FPR."""
    df_ord = df.sort_values(["f1", "fpr"], ascending=[False, True])
    return str(df_ord.iloc[0]["algoritmo"])