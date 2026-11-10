"""Detector por firmas (baseline clasico) para comparar con el NBM.

Implementa reglas estaticas tipicas de un IDS basado en firmas (estilo
Snort/Suricata). No aprende nada: clasifica un evento como anomalo si
dispara AL MENOS UNA de las reglas predefinidas.

Reglas implementadas (las mas comunes en IDS comerciales):
- R1: intentos_fallidos > 10           -> fuerza bruta
- R2: errores_4xx > 50                 -> escaneo
- R3: bytes_transferidos_mb > 100      -> exfiltracion
- R4: hora < 6 o hora > 22             -> acceso nocturno
- R5: usuarios_distintos_ip > 10       -> password spraying
- R6: accesos_rutas_sensibles > 20     -> escalada de privilegios
- R7: num_peticiones_hora > 300        -> posible DoS
- R8: duracion_sesion_min > 180        -> sesion anormalmente larga

Uso:
    python -m app.nbm.signature_detector
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
)

from app.nbm import config
from pathlib import Path

# ---------------------------------------------------------------------------
# Reglas estaticas (firmas)
# ---------------------------------------------------------------------------
def _firma_fuerza_bruta(row: dict) -> bool:
    return row.get("intentos_fallidos", 0) > 10


def _firma_escaneo(row: dict) -> bool:
    return row.get("errores_4xx", 0) > 50


def _firma_exfiltracion(row: dict) -> bool:
    return row.get("bytes_transferidos_mb", 0) > 100


def _firma_nocturno(row: dict) -> bool:
    hora = row.get("hora", 12)
    return hora < 6 or hora > 22


def _firma_password_spraying(row: dict) -> bool:
    return row.get("usuarios_distintos_ip", 1) > 10


def _firma_escalada(row: dict) -> bool:
    return row.get("accesos_rutas_sensibles", 0) > 20


def _firma_dos(row: dict) -> bool:
    return row.get("num_peticiones_hora", 0) > 300


def _firma_sesion_larga(row: dict) -> bool:
    return row.get("duracion_sesion_min", 0) > 180


FIRMAS = [
    ("R1_fuerza_bruta", _firma_fuerza_bruta),
    ("R2_escaneo", _firma_escaneo),
    ("R3_exfiltracion", _firma_exfiltracion),
    ("R4_nocturno", _firma_nocturno),
    ("R5_password_spraying", _firma_password_spraying),
    ("R6_escalada", _firma_escalada),
    ("R7_dos", _firma_dos),
    ("R8_sesion_larga", _firma_sesion_larga),
]


@dataclass
class ResultadoFirmas:
    y_pred: np.ndarray
    firmas_disparadas: dict[str, int]
    tiempo_prediccion_s: float


def predecir_con_firmas(df_test: pd.DataFrame) -> ResultadoFirmas:
    """Aplica todas las firmas a cada evento. Devuelve 1 si al menos una dispara."""
    t0 = time.perf_counter()

    y_pred = []
    disparos = {nombre: 0 for nombre, _ in FIRMAS}

    for _, row in df_test.iterrows():
        row_dict = row.to_dict()
        disparada = False
        for nombre, fn in FIRMAS:
            if fn(row_dict):
                disparos[nombre] += 1
                disparada = True
        y_pred.append(1 if disparada else 0)

    t_pred = time.perf_counter() - t0
    return ResultadoFirmas(
        y_pred=np.array(y_pred),
        firmas_disparadas=disparos,
        tiempo_prediccion_s=t_pred,
    )


def _evaluar(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "fpr": round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def main():
    print("[firmas] Cargando dataset...")
    df = pd.read_csv(config.DATASET_CSV)
    print(f"      Registros: {len(df)} | Anomalias: {int(df['is_anomalous'].sum())}")

    # Mismo split que el NBM (mismo random_state) para comparacion justa
    _, df_test, _, y_test = train_test_split(
        df, df["is_anomalous"].astype(int).to_numpy(),
        test_size=config.TEST_SIZE, random_state=config.SEED, stratify=df["is_anomalous"],
    )
    print(f"      Test: {len(df_test)} ({int(y_test.sum())} anomalias reales)")

    print("[firmas] Aplicando reglas estaticas...")
    res = predecir_con_firmas(df_test)
    metricas = _evaluar(y_test, res.y_pred)

    # Comparar con el NBM (OneClassSVM) cargando el CSV de comparativa
    print("[firmas] Cargando resultados del NBM para comparar...")
    try:
        nbm_df = pd.read_csv(config.COMPARATIVA_CSV)
        ocsvm = nbm_df[nbm_df["algoritmo"] == "OneClassSVM"].iloc[0]
        nbm_metricas = {
            "precision": float(ocsvm["precision"]),
            "recall": float(ocsvm["recall"]),
            "f1": float(ocsvm["f1"]),
            "fpr": float(ocsvm["fpr"]),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"      [WARN] No se pudo cargar comparativa NBM: {exc}")
        nbm_metricas = None

    # --- Imprimir tabla comparativa ---
    print("\n" + "=" * 90)
    print("COMPARACION: DETECCION POR FIRMAS vs NBM (OneClassSVM)")
    print("=" * 90)
    print(f"{'Metrica':<20} {'Firmas':>12} {'NBM (OC-SVM)':>15} {'Diferencia':>15}")
    print("-" * 90)
    for k in ["precision", "recall", "f1", "fpr"]:
        firmas_v = metricas[k]
        if nbm_metricas:
            nbm_v = nbm_metricas[k]
            diff = f"{(nbm_v - firmas_v):+.4f}"
        else:
            nbm_v = "N/A"
            diff = "N/A"
        print(f"{k:<20} {firmas_v:>12.4f} {str(nbm_v):>15} {diff:>15}")
    print("-" * 90)
    print(f"TP={metricas['tp']}  TN={metricas['tn']}  FP={metricas['fp']}  FN={metricas['fn']}")
    print(f"Tiempo de prediccion: {res.tiempo_prediccion_s*1000:.1f} ms")
    print("=" * 90)

    # --- Firmas disparadas ---
    print("\nFIRMAS DISPARADAS EN EL TEST SET")
    print("-" * 90)
    for nombre, cuenta in res.firmas_disparadas.items():
        if cuenta > 0:
            print(f"  {nombre:<25} {cuenta:>5} eventos")

    # --- Guardar CSV ---

    VALIDATION_RESULTS = Path(__file__).resolve().parent / "results"
    VALIDATION_RESULTS.mkdir(exist_ok=True)
    ruta = VALIDATION_RESULTS / "comparativa_firmas_vs_nbm.csv"
    fila_firmas = {"algoritmo": "Firmas", **metricas}
    fila_nbm = {"algoritmo": "OneClassSVM", **(nbm_metricas or {})}
    pd.DataFrame([fila_firmas, fila_nbm]).to_csv(ruta, index=False)
    print(f"\n[firmas] Resultados en {ruta}")


if __name__ == "__main__":
    main()