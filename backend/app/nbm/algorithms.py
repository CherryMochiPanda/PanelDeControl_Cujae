"""Implementaciones de los algoritmos de deteccion de anomalias.

- Isolation Forest (sklearn)
- One-Class SVM (sklearn)
- Autoencoder (PyTorch, import diferido)

Todas exponen la misma interfaz: fit(X_train, y_train) / predict(X_test)
devolviendo 0/1 (0=normal, 1=anomalia).

Incluye medicion de consumo de recursos:
- CPU: porcentaje de la maquina (0-100%), calculado como
  (tiempo_cpu_proceso / tiempo_pared) / n_cores * 100.
- RAM: incremento de RSS (MB) durante la operacion, con muestreo
  en hilo para captar el pico.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from app.nbm import config


@dataclass
class ResultadoAlgoritmo:
    nombre: str
    hiperparametros: dict[str, Any]
    y_pred: np.ndarray
    tiempo_entrenamiento_s: float
    tiempo_prediccion_s: float
    cpu_entrenamiento_pct: float = 0.0
    cpu_prediccion_pct: float = 0.0
    ram_entrenamiento_mb: float = 0.0
    ram_prediccion_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _medir_recursos(func, *args, **kwargs):
    """Ejecuta func y devuelve (resultado, tiempo_s, cpu_pct_maquina, ram_mb_incremento).

    - cpu_pct_maquina: % de la maquina completa (0-100). Se calcula con
      tiempo de CPU del proceso (user+system) dividido entre tiempo de pared
      y normalizado por cantidad de nucleos logicos.
    - ram_mb_incremento: (RSS_pico - RSS_antes) en MB. Muestrea RSS cada 10ms
      en un hilo para captar el pico durante la ejecucion.
    """
    proc = psutil.Process(os.getpid())
    n_cores = psutil.cpu_count(logical=True) or 1

    ram_antes = proc.memory_info().rss
    ram_pico = [ram_antes]
    stop = threading.Event()

    def _muestrear():
        while not stop.is_set():
            try:
                actual = proc.memory_info().rss
                if actual > ram_pico[0]:
                    ram_pico[0] = actual
            except psutil.Error:
                pass
            time.sleep(0.01)

    hilo = threading.Thread(target=_muestrear, daemon=True)
    hilo.start()

    t_cpu_antes = proc.cpu_times()
    t0 = time.perf_counter()
    resultado = func(*args, **kwargs)
    tiempo = time.perf_counter() - t0
    t_cpu_despues = proc.cpu_times()

    stop.set()
    hilo.join(timeout=0.5)

    cpu_seg = (t_cpu_despues.user - t_cpu_antes.user) + (t_cpu_despues.system - t_cpu_antes.system)
    cpu_pct_nucleo = (cpu_seg / tiempo) * 100 if tiempo > 0 else 0.0
    cpu_pct_maquina = cpu_pct_nucleo / n_cores

    ram_mb = max(0.0, (ram_pico[0] - ram_antes) / (1024 * 1024))
    return resultado, tiempo, cpu_pct_maquina, ram_mb


def ejecutar_isolation_forest(X_train, y_train, X_test) -> ResultadoAlgoritmo:
    """Entrena IF solo con datos normales y predice sobre test."""
    X_train_normal = X_train[y_train == 0]
    params = {
        "n_estimators": config.IF_N_ESTIMATORS,
        "contamination": config.IF_CONTAMINATION,
        "random_state": config.SEED,
    }

    model, t_fit, cpu_fit, ram_fit = _medir_recursos(
        lambda: IsolationForest(**params).fit(X_train_normal)
    )
    raw, t_pred, cpu_pred, ram_pred = _medir_recursos(model.predict, X_test)
    y_pred = np.where(raw == -1, 1, 0)

    return ResultadoAlgoritmo(
        nombre="IsolationForest", hiperparametros=params,
        y_pred=y_pred, tiempo_entrenamiento_s=t_fit, tiempo_prediccion_s=t_pred,
        cpu_entrenamiento_pct=cpu_fit, cpu_prediccion_pct=cpu_pred,
        ram_entrenamiento_mb=ram_fit, ram_prediccion_mb=ram_pred,
    )


def ejecutar_one_class_svm(X_train, y_train, X_test) -> ResultadoAlgoritmo:
    """Entrena OC-SVM solo con datos normales."""
    X_train_normal = X_train[y_train == 0]
    params = {
        "kernel": config.OCSVM_KERNEL,
        "gamma": config.OCSVM_GAMMA,
        "nu": config.OCSVM_NU,
    }

    model, t_fit, cpu_fit, ram_fit = _medir_recursos(
        lambda: OneClassSVM(**params).fit(X_train_normal)
    )
    raw, t_pred, cpu_pred, ram_pred = _medir_recursos(model.predict, X_test)
    y_pred = np.where(raw == -1, 1, 0)

    return ResultadoAlgoritmo(
        nombre="OneClassSVM", hiperparametros=params,
        y_pred=y_pred, tiempo_entrenamiento_s=t_fit, tiempo_prediccion_s=t_pred,
        cpu_entrenamiento_pct=cpu_fit, cpu_prediccion_pct=cpu_pred,
        ram_entrenamiento_mb=ram_fit, ram_prediccion_mb=ram_pred,
    )


def ejecutar_autoencoder(X_train, y_train, X_test) -> ResultadoAlgoritmo:
    """Autoencoder con PyTorch. Import diferido: falla con mensaje claro si no hay torch."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "Autoencoder requiere torch. Instala con: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    torch.manual_seed(config.SEED)
    X_train_normal = X_train[y_train == 0].astype(np.float32)
    X_test_f = X_test.astype(np.float32)
    n_features = X_train_normal.shape[1]

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 16), nn.ReLU(),
                nn.Linear(16, config.AE_HIDDEN), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(config.AE_HIDDEN, 16), nn.ReLU(),
                nn.Linear(16, n_features),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = AE()
    optim = torch.optim.Adam(model.parameters(), lr=config.AE_LR)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_normal)),
        batch_size=config.AE_BATCH_SIZE, shuffle=True,
    )

    params = {
        "epochs": config.AE_EPOCHS, "batch_size": config.AE_BATCH_SIZE,
        "lr": config.AE_LR, "hidden": config.AE_HIDDEN,
        "umbral_percentil": config.AE_UMBRAL_PERCENTIL,
    }

    def _entrenar():
        model.train()
        for _ in range(config.AE_EPOCHS):
            for (batch,) in loader:
                optim.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optim.step()
        return model

    model, t_fit, cpu_fit, ram_fit = _medir_recursos(_entrenar)

    def _predecir():
        model.eval()
        with torch.no_grad():
            # Warm-up: la primera inferencia en PyTorch lanza hilos internos
            # (intra-op parallelism) y compila kernels JIT. Si se mide, contamina
            # las metricas de CPU/RAM (aparecen valores >100%). Se ejecuta con
            # lotes pequeños y NO se mide.
            _ = model(torch.from_numpy(X_train_normal[:16]))
            _ = model(torch.from_numpy(X_test_f[:16]))

            err_train = ((model(torch.from_numpy(X_train_normal)) - torch.from_numpy(X_train_normal)) ** 2).mean(dim=1).numpy()
            umbral = float(np.percentile(err_train, config.AE_UMBRAL_PERCENTIL))
            err_test = ((model(torch.from_numpy(X_test_f)) - torch.from_numpy(X_test_f)) ** 2).mean(dim=1).numpy()
            return (err_test > umbral).astype(int), umbral
    (y_pred, umbral), t_pred, cpu_pred, ram_pred = _medir_recursos(_predecir)

    return ResultadoAlgoritmo(
        nombre="Autoencoder", hiperparametros=params,
        y_pred=y_pred, tiempo_entrenamiento_s=t_fit, tiempo_prediccion_s=t_pred,
        cpu_entrenamiento_pct=cpu_fit, cpu_prediccion_pct=cpu_pred,
        ram_entrenamiento_mb=ram_fit, ram_prediccion_mb=ram_pred,
        extra={"umbral_reconstruccion": umbral},
    )