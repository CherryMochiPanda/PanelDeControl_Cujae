"""Implementaciones de los algoritmos de deteccion de anomalias.

- Isolation Forest (sklearn)
- One-Class SVM (sklearn)
- Autoencoder (PyTorch, import diferido)

Todas exponen la misma interfaz: fit(X_train, y_train) / predict(X_test)
devolviendo 0/1 (0=normal, 1=anomalia).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
    extra: dict[str, Any] = field(default_factory=dict)


def ejecutar_isolation_forest(X_train, y_train, X_test) -> ResultadoAlgoritmo:
    """Entrena IF solo con datos normales y predice sobre test."""
    X_train_normal = X_train[y_train == 0]
    params = {
        "n_estimators": config.IF_N_ESTIMATORS,
        "contamination": config.IF_CONTAMINATION,
        "random_state": config.SEED,
    }
    t0 = time.perf_counter()
    model = IsolationForest(**params).fit(X_train_normal)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw = model.predict(X_test)
    t_pred = time.perf_counter() - t0

    y_pred = np.where(raw == -1, 1, 0)
    return ResultadoAlgoritmo(
        nombre="IsolationForest", hiperparametros=params,
        y_pred=y_pred, tiempo_entrenamiento_s=t_fit, tiempo_prediccion_s=t_pred,
    )


def ejecutar_one_class_svm(X_train, y_train, X_test) -> ResultadoAlgoritmo:
    """Entrena OC-SVM solo con datos normales."""
    X_train_normal = X_train[y_train == 0]
    params = {
        "kernel": config.OCSVM_KERNEL,
        "gamma": config.OCSVM_GAMMA,
        "nu": config.OCSVM_NU,
    }
    t0 = time.perf_counter()
    model = OneClassSVM(**params).fit(X_train_normal)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw = model.predict(X_test)
    t_pred = time.perf_counter() - t0

    y_pred = np.where(raw == -1, 1, 0)
    return ResultadoAlgoritmo(
        nombre="OneClassSVM", hiperparametros=params,
        y_pred=y_pred, tiempo_entrenamiento_s=t_fit, tiempo_prediccion_s=t_pred,
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

    t0 = time.perf_counter()
    model.train()
    for _ in range(config.AE_EPOCHS):
        for (batch,) in loader:
            optim.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optim.step()
    t_fit = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        err_train = ((model(torch.from_numpy(X_train_normal)) - torch.from_numpy(X_train_normal)) ** 2).mean(dim=1).numpy()
        umbral = float(np.percentile(err_train, config.AE_UMBRAL_PERCENTIL))

        t0 = time.perf_counter()
        err_test = ((model(torch.from_numpy(X_test_f)) - torch.from_numpy(X_test_f)) ** 2).mean(dim=1).numpy()
        t_pred = time.perf_counter() - t0

    y_pred = (err_test > umbral).astype(int)
    return ResultadoAlgoritmo(
        nombre="Autoencoder", hiperparametros=params,
        y_pred=y_pred, tiempo_entrenamiento_s=t_fit, tiempo_prediccion_s=t_pred,
        extra={"umbral_reconstruccion": umbral},
    )