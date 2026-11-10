"""Simulador de escenarios de ataque para validar el NBM en produccion.

Genera secuencias realistas de ataques (no eventos aislados), las inserta
en auditoria.security_logs (anonimizadas) y evalua si el modelo cargado
las detecta correctamente.

Tipos de ataque simulados:
1. Fuerza bruta: multiples login_fail + 1 login exitoso desde misma IP.
2. Escaneo de rutas: multiples 404 en rutas distintas.
3. Exfiltracion: descargas grandes consecutivas.
4. Movimiento lateral: accesos a rutas sensibles desde usuario comprometido.
5. Password spraying: muchos usuarios con 1 intento fallido desde misma IP.
6. Acceso nocturno: login exitoso a horas no laborales.

Uso:
    python -m app.nbm.attack_simulator
    python -m app.nbm.attack_simulator --secuencias 30
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.database import SessionLocal
from app.nbm import config, detection_service
from app.services.log_service import write_security_log

RESULTS_DIR = config.RESULTS_DIR / "attack_simulation"
RESULTS_DIR.mkdir(exist_ok=True)

# Usuarios ficticios (mismos que usa el dataset sintetico)
USUARIOS = [f"user_{i:04d}" for i in range(50)]
ROLES = ["estudiante", "profesor", "jefe_proceso", "directivo"]
COMPONENTES = ["dashboard", "api_auth", "api_indicadores", "api_export", "api_modelos"]


def _evento_base(
    *, usuario: str, rol: str, componente: str, hora: int, dia: int,
    ip: str = "", duracion: float = 10.0, peticiones: int = 5,
    intentos: int = 0, err4xx: int = 0, err5xx: int = 0,
    bytes_mb: float = 2.0, sensibles: int = 0, ips_distintas: int = 1,
    protocolo: str = "HTTPS",
) -> dict:
    """Construye un evento con las features que espera /detectar."""
    return {
        "usuario": usuario, "rol": rol, "componente": componente,
        "hora": hora, "dia_semana": dia,
        "es_horario_laboral": 1 if 8 <= hora < 19 else 0,
        "duracion_sesion_min": duracion,
        "num_peticiones_hora": peticiones,
        "intentos_fallidos": intentos,
        "errores_4xx": err4xx, "errores_5xx": err5xx,
        "bytes_transferidos_mb": bytes_mb,
        "accesos_rutas_sensibles": sensibles,
        "usuarios_distintos_ip": ips_distintas,
        "ip_externa": 0 if ip.startswith("192.168.") else 1,
        "protocolo": protocolo,
        "_ip_raw": ip,  # solo para el log, no se envia al modelo
        "_anomaly_type": "",  # se rellena segun el ataque
    }


def _secuencia_fuerza_bruta(rng: random.Random) -> list[dict]:
    """Multiples login_fail + 1 login exitoso desde misma IP externa."""
    usuario = rng.choice(USUARIOS)
    ip = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    hora = rng.choice([2, 3, 4, 23])
    eventos = []
    for i in range(rng.randint(8, 15)):
        ev = _evento_base(
            usuario=usuario, rol="estudiante", componente="api_auth",
            hora=hora, dia=rng.randint(0, 4), ip=ip,
            intentos=rng.randint(1, 3), peticiones=rng.randint(5, 15),
        )
        ev["_anomaly_type"] = "fuerza_bruta"
        eventos.append(ev)
    return eventos


def _secuencia_escaneo(rng: random.Random) -> list[dict]:
    """Multiples 404 en rutas distintas desde misma IP."""
    usuario = rng.choice(USUARIOS)
    ip = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    hora = rng.randint(8, 18)
    eventos = []
    for i in range(rng.randint(15, 30)):
        ev = _evento_base(
            usuario=usuario, rol="estudiante", componente="api_export",
            hora=hora, dia=rng.randint(0, 4), ip=ip,
            err4xx=rng.randint(3, 8), peticiones=rng.randint(20, 50),
            sensibles=rng.randint(1, 3),
        )
        ev["_anomaly_type"] = "escaneo"
        eventos.append(ev)
    return eventos


def _secuencia_exfiltracion(rng: random.Random) -> list[dict]:
    """Descargas grandes consecutivas en sesion larga."""
    usuario = rng.choice(USUARIOS)
    hora = rng.randint(9, 17)
    eventos = []
    for i in range(rng.randint(4, 8)):
        ev = _evento_base(
            usuario=usuario, rol="profesor", componente="api_export",
            hora=hora, dia=rng.randint(0, 4), ip="192.168.1.45",
            bytes_mb=round(rng.uniform(80, 200), 2),
            duracion=rng.uniform(30, 120), peticiones=rng.randint(3, 10),
        )
        ev["_anomaly_type"] = "exfiltracion"
        eventos.append(ev)
    return eventos


def _secuencia_movimiento_lateral(rng: random.Random) -> list[dict]:
    """Accesos a rutas sensibles + componente api_modelos + hora nocturna."""
    usuario = rng.choice(USUARIOS)
    hora = rng.choice([1, 2, 3, 4])
    eventos = []
    for i in range(rng.randint(6, 12)):
        ev = _evento_base(
            usuario=usuario, rol="jefe_proceso", componente="api_modelos",
            hora=hora, dia=rng.randint(0, 4), ip="10.0.0.15",
            intentos=rng.randint(1, 3), sensibles=rng.randint(2, 5),
            ips_distintas=rng.randint(2, 4), peticiones=rng.randint(5, 20),
        )
        ev["_anomaly_type"] = "movimiento_lateral"
        eventos.append(ev)
    return eventos


def _secuencia_password_spraying(rng: random.Random) -> list[dict]:
    """Muchos usuarios distintos, 1 intento fallido cada uno, misma IP."""
    ip = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    hora = rng.randint(9, 17)
    usuarios = rng.sample(USUARIOS, rng.randint(10, 20))
    eventos = []
    for u in usuarios:
        ev = _evento_base(
            usuario=u, rol="estudiante", componente="api_auth",
            hora=hora, dia=rng.randint(0, 4), ip=ip,
            intentos=1, peticiones=rng.randint(2, 8),
            ips_distintas=len(usuarios),
        )
        ev["_anomaly_type"] = "password_spraying"
        eventos.append(ev)
    return eventos


def _secuencia_nocturno(rng: random.Random) -> list[dict]:
    """Login exitoso a horas no laborales (usuario legitimo comprometido)."""
    usuario = rng.choice(USUARIOS)
    hora = rng.choice([0, 1, 2, 3, 4, 22, 23])
    eventos = []
    for i in range(rng.randint(3, 6)):
        ev = _evento_base(
            usuario=usuario, rol="directivo", componente="dashboard",
            hora=hora, dia=rng.randint(0, 4), ip="192.168.2.10",
            peticiones=rng.randint(3, 10),
        )
        ev["_anomaly_type"] = "nocturno"
        eventos.append(ev)
    return eventos


GENERADORES = {
    "fuerza_bruta": _secuencia_fuerza_bruta,
    "escaneo": _secuencia_escaneo,
    "exfiltracion": _secuencia_exfiltracion,
    "movimiento_lateral": _secuencia_movimiento_lateral,
    "password_spraying": _secuencia_password_spraying,
    "nocturno": _secuencia_nocturno,
}


async def _persistir_en_logs(secuencias: list[tuple[str, list[dict]]]) -> None:
    """Inserta los eventos de cada secuencia en auditoria.security_logs."""
    async with SessionLocal() as session:
        for tipo, eventos in secuencias:
            for ev in eventos:
                await write_security_log(
                    session,
                    event_type="login" if tipo == "fuerza_bruta" else "indicator_read",
                    severity="WARNING" if ev.get("_anomaly_type") else "INFO",
                    user={"user_id": "00000000-0000-0000-0000-000000000000",
                          "username": ev["usuario"], "role": ev["rol"]},
                    ip=ev["_ip_raw"],
                    operation="login" if tipo == "fuerza_bruta" else "read",
                    resource_type=ev["componente"],
                    details={"simulated_attack": tipo},
                    commit=False,
                )
        await session.commit()


def _evaluar_secuencias(secuencias: list[tuple[str, list[dict]]]) -> pd.DataFrame:
    """Evalua cada secuencia con el modelo cargado y devuelve tabla de resultados."""
    filas = []
    for tipo, eventos in secuencias:
        # Quitar campos internos que no son features
        eventos_limpios = [
            {k: v for k, v in e.items() if not k.startswith("_")}
            for e in eventos
        ]
        try:
            resultados = detection_service.detectar_eventos(eventos_limpios)
        except RuntimeError as exc:
            print(f"[FATAL] {exc}")
            sys.exit(1)

        detectados = sum(1 for r in resultados if r["is_anomaly"])
        scores = [r["score"] for r in resultados]
        filas.append({
            "tipo_ataque": tipo,
            "n_eventos": len(eventos),
            "detectados": detectados,
            "tasa_deteccion": round(detectados / len(eventos), 4),
            "score_min": round(min(scores), 4),
            "score_max": round(max(scores), 4),
            "score_promedio": round(sum(scores) / len(scores), 4),
            "secuencia_detectada": detectados > 0,
        })
    return pd.DataFrame(filas)


async def main(n_secuencias: int) -> int:
    rng = random.Random(42)
    tipos = list(GENERADORES.keys())

    print(f"[1/4] Generando {n_secuencias} secuencias de ataque...")
    secuencias: list[tuple[str, list[dict]]] = []
    for i in range(n_secuencias):
        tipo = tipos[i % len(tipos)]
        eventos = GENERADORES[tipo](rng)
        secuencias.append((tipo, eventos))
    total_eventos = sum(len(e) for _, e in secuencias)
    print(f"      {n_secuencias} secuencias, {total_eventos} eventos totales")

    print("[2/4] Insertando en auditoria.security_logs (anonimizado)...")
    await _persistir_en_logs(secuencias)
    print(f"      {total_eventos} logs insertados")

    print("[3/4] Evaluando con el modelo cargado...")
    df = _evaluar_secuencias(secuencias)

    print("[4/4] Guardando reporte...")
    ruta = RESULTS_DIR / f"attack_simulation_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(ruta, index=False)
    print(f"      Reporte en {ruta}\n")

    print("=" * 90)
    print("RESULTADOS DE LA SIMULACION DE ATAQUES - NBM CUJAE")
    print("=" * 90)
    cols = ["tipo_ataque", "n_eventos", "detectados", "tasa_deteccion",
            "score_promedio", "secuencia_detectada"]
    print(df[cols].to_string(index=False))
    print("-" * 90)

    # Resumen por tipo de ataque
    print("\nRESUMEN POR TIPO DE ATAQUE")
    print("-" * 90)
    resumen = df.groupby("tipo_ataque").agg({
        "n_eventos": "sum",
        "detectados": "sum",
        "tasa_deteccion": "mean",
        "secuencia_detectada": "sum",
    }).round(4)
    resumen["n_secuencias"] = df.groupby("tipo_ataque").size()
    print(resumen.to_string())

    # Metricas globales
    total_sec = len(df)
    sec_detectadas = int(df["secuencia_detectada"].sum())
    tasa_global = df["detectados"].sum() / df["n_eventos"].sum()
    print("-" * 90)
    print(f"Secuencias detectadas: {sec_detectadas}/{total_sec} ({sec_detectadas/total_sec*100:.1f}%)")
    print(f"Eventos detectados: {df['detectados'].sum()}/{df['n_eventos'].sum()} ({tasa_global*100:.1f}%)")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador de ataques para validar el NBM.")
    parser.add_argument("--secuencias", type=int, default=30, help="Numero de secuencias a generar.")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.secuencias)))