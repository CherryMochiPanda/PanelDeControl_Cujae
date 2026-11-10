"""Comparacion de detectores sobre ataques STEALTH (sigilosos).

Genera ataques que NO disparan ninguna de las 8 reglas de firma (cada
feature individual esta por debajo de su umbral), pero cuya combinacion
es anomala. Sirve para demostrar que:
- El detector por firmas falla con ataques desconocidos/evasivos.
- El NBM (OneClassSVM) SI los detecta porque modelo la distribucion conjunta.

Tipos de ataque stealth:
1. Exfiltracion gota a gota: bytes=85 (<100), duracion=170 (<180), peticiones=8.
2. Movimiento lateral silencioso: accesos_sensibles=18 (<20), ips=9 (<10), hora=14.
3. Password spraying distribuido: ips=9 (<10), intentos=9 (<10), componente=api_auth.

Uso:
    python -m app.validation.stealth_attacks
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from app.nbm import config, detection_service
from app.validation.signature_detector import predecir_con_firmas


VALIDATION_RESULTS = Path(__file__).resolve().parent / "results"
VALIDATION_RESULTS.mkdir(exist_ok=True)

N_NORMALES = 200
N_STEALTH_POR_TIPO = 33  # 33 * 3 = 99 stealth


def _evento_normal(row: pd.Series) -> dict:
    """Convierte una fila del dataset sintetico a dict de EventoIn."""
    return {
        "hora": int(row["hora"]),
        "dia_semana": int(row["dia_semana"]),
        "es_horario_laboral": int(row["es_horario_laboral"]),
        "duracion_sesion_min": float(row["duracion_sesion_min"]),
        "num_peticiones_hora": int(row["num_peticiones_hora"]),
        "intentos_fallidos": int(row["intentos_fallidos"]),
        "errores_4xx": int(row["errores_4xx"]),
        "errores_5xx": int(row["errores_5xx"]),
        "bytes_transferidos_mb": float(row["bytes_transferidos_mb"]),
        "accesos_rutas_sensibles": int(row["accesos_rutas_sensibles"]),
        "usuarios_distintos_ip": int(row["usuarios_distintos_ip"]),
        "ip_externa": int(not str(row["ip_origen"]).startswith("192.168.")),
        "rol": str(row["rol"]),
        "componente": str(row["componente"]),
        "protocolo": str(row["protocolo"]),
    }


def _exfiltracion_stealth(rng: random.Random) -> dict:
    """Bytes y duracion altos, pero por debajo de los umbrales de R3 y R8."""
    return {
        "hora": rng.randint(9, 17),
        "dia_semana": rng.randint(0, 4),
        "es_horario_laboral": 1,
        "duracion_sesion_min": round(rng.uniform(150, 178), 2),   # < 180 (R8)
        "num_peticiones_hora": rng.randint(5, 15),                 # < 300 (R7)
        "intentos_fallidos": 0,
        "errores_4xx": 0,
        "errores_5xx": 0,
        "bytes_transferidos_mb": round(rng.uniform(75, 98), 2),    # < 100 (R3)
        "accesos_rutas_sensibles": 0,
        "usuarios_distintos_ip": 1,
        "ip_externa": 0,
        "rol": "profesor",
        "componente": "api_export",
        "protocolo": "HTTPS",
    }


def _movimiento_lateral_stealth(rng: random.Random) -> dict:
    """Accesos sensibles e IPs altos, pero por debajo de R6 y R5."""
    return {
        "hora": rng.randint(9, 17),                                # no R4
        "dia_semana": rng.randint(0, 4),
        "es_horario_laboral": 1,
        "duracion_sesion_min": round(rng.uniform(20, 50), 2),
        "num_peticiones_hora": rng.randint(10, 40),
        "intentos_fallidos": rng.randint(2, 5),                    # < 10 (R1)
        "errores_4xx": rng.randint(0, 3),
        "errores_5xx": 0,
        "bytes_transferidos_mb": round(rng.uniform(3, 10), 2),
        "accesos_rutas_sensibles": rng.randint(12, 19),            # < 20 (R6)
        "usuarios_distintos_ip": rng.randint(6, 9),                # < 10 (R5)
        "ip_externa": 1,
        "rol": "jefe_proceso",
        "componente": "api_modelos",
        "protocolo": "HTTPS",
    }


def _password_spray_stealth(rng: random.Random) -> dict:
    """Muchas IPs e intentos moderados, pero por debajo de R5 y R1."""
    return {
        "hora": rng.randint(9, 17),                                # no R4
        "dia_semana": rng.randint(0, 4),
        "es_horario_laboral": 1,
        "duracion_sesion_min": round(rng.uniform(5, 15), 2),
        "num_peticiones_hora": rng.randint(3, 10),
        "intentos_fallidos": rng.randint(6, 9),                    # < 10 (R1)
        "errores_4xx": rng.randint(0, 2),
        "errores_5xx": 0,
        "bytes_transferidos_mb": round(rng.uniform(1, 3), 2),
        "accesos_rutas_sensibles": 0,
        "usuarios_distintos_ip": rng.randint(6, 9),                # < 10 (R5)
        "ip_externa": 1,
        "rol": "estudiante",
        "componente": "api_auth",
        "protocolo": "HTTPS",
    }


GENERADORES_STEALTH = {
    "exfiltracion_stealth": _exfiltracion_stealth,
    "movimiento_lateral_stealth": _movimiento_lateral_stealth,
    "password_spray_stealth": _password_spray_stealth,
}


def main() -> int:
    rng = random.Random(42)

    print("[stealth] Cargando dataset para extraer eventos normales...")
    df = pd.read_csv(config.DATASET_CSV)
    df_normales = df[df["is_anomalous"] == 0].sample(n=N_NORMALES, random_state=config.SEED)

    print(f"[stealth] Generando {N_STEALTH_POR_TIPO} x 3 tipos de ataques stealth...")
    eventos_stealth: list[dict] = []
    for tipo, gen in GENERADORES_STEALTH.items():
        for _ in range(N_STEALTH_POR_TIPO):
            ev = gen(rng)
            eventos_stealth.append(ev)

    # --- Construir el dataset de evaluacion ---
    filas = []
    for _, row in df_normales.iterrows():
        filas.append({"evento": _evento_normal(row), "y_true": 0})
    for ev in eventos_stealth:
        filas.append({"evento": ev, "y_true": 1})

    df_eval = pd.DataFrame(filas).sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    y_true = df_eval["y_true"].to_numpy()

    print(f"[stealth] Total evaluado: {len(df_eval)} ({int(y_true.sum())} stealth, "
          f"{int((1 - y_true).sum())} normales)")

    # --- 1. Evaluar firmas ---
    print("[stealth] Evaluando detector por FIRMAS...")
    df_solo_features = pd.DataFrame([e for e in df_eval["evento"]])
    res_firmas = predecir_con_firmas(df_solo_features)

    # --- 2. Evaluar NBM ---
    print("[stealth] Evaluando NBM (OneClassSVM)...")
    eventos_limpios = [dict(e) for e in df_eval["evento"]]
    res_nbm = detection_service.detectar_eventos(eventos_limpios)
    y_pred_nbm = np.array([1 if r["is_anomaly"] else 0 for r in res_nbm])

    # --- 3. Metricas ---
    def _metricas(y_t, y_p):
        from sklearn.metrics import f1_score, precision_score, recall_score
        return {
            "precision": precision_score(y_t, y_p, zero_division=0),
            "recall": recall_score(y_t, y_p, zero_division=0),
            "f1": f1_score(y_t, y_p, zero_division=0),
        }

    m_firmas = _metricas(y_true, res_firmas.y_pred)
    m_nbm = _metricas(y_true, y_pred_nbm)

    # --- 4. Reportar ---
    print("\n" + "=" * 90)
    print("COMPARACION SOBRE ATAQUES STEALTH (evasivos)")
    print("=" * 90)
    print(f"{'Detector':<20} {'Precision':>12} {'Recall':>12} {'F1':>12} {'TP':>6} {'FN':>6}")
    print("-" * 90)
    tp_f = int(((y_true == 1) & (res_firmas.y_pred == 1)).sum())
    fn_f = int(((y_true == 1) & (res_firmas.y_pred == 0)).sum())
    tp_n = int(((y_true == 1) & (y_pred_nbm == 1)).sum())
    fn_n = int(((y_true == 1) & (y_pred_nbm == 0)).sum())
    print(f"{'Firmas':<20} {m_firmas['precision']:>12.4f} {m_firmas['recall']:>12.4f} {m_firmas['f1']:>12.4f} {tp_f:>6} {fn_f:>6}")
    print(f"{'NBM (OC-SVM)':<20} {m_nbm['precision']:>12.4f} {m_nbm['recall']:>12.4f} {m_nbm['f1']:>12.4f} {tp_n:>6} {fn_n:>6}")
    print("=" * 90)

    # --- 5. Desglose por tipo de ataque stealth ---
    print("\nDETECCION POR TIPO DE ATAQUE STEALTH")
    print("-" * 90)
    print(f"{'Tipo':<30} {'N':>5} {'Firmas TP':>10} {'NBM TP':>10}")
    print("-" * 90)
    for tipo, gen in GENERADORES_STEALTH.items():
        # Buscar los eventos de este tipo en df_eval
        # Los identificamos por tener el componente y horas caracteristicas
        # (más simple: regenerar y comparar con los índices)
        pass
    # En su lugar, resumimos globalmente
    print(f"{'Total stealth':<30} {int(y_true.sum()):>5} {tp_f:>10} {tp_n:>10}")
    print("=" * 90)

    # --- 6. Guardar CSV ---
    ruta = VALIDATION_RESULTS / "stealth_comparison.csv"
    pd.DataFrame([
        {"detector": "Firmas", **m_firmas, "tp": tp_f, "fn": fn_f},
        {"detector": "OneClassSVM", **m_nbm, "tp": tp_n, "fn": fn_n},
    ]).to_csv(ruta, index=False)
    print(f"\n[stealth] Resultados en {ruta}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())