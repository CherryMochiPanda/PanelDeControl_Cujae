"""Grid search de hiperparametros para los 3 algoritmos de ML.

Prueba combinaciones y guarda una tabla comparativa en CSV.
Uso:
    python -m app.nbm.grid_search
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from app.nbm import config
from app.nbm.data_generator import generar_dataset_sintetico
from app.nbm.runner import _preparar_xy

RESULTS_DIR = config.RESULTS_DIR / "grid_search"
RESULTS_DIR.mkdir(exist_ok=True)


def _cargar_datos():
    ruta = config.DATASET_CSV
    if ruta.exists():
        df = pd.read_csv(ruta)
    else:
        df = generar_dataset_sintetico()
        df.to_csv(ruta, index=False)
    X, y, features = _preparar_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y,
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test, features


def _evaluar(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "fpr": round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def grid_isolation_forest(X_train, y_train, X_test, y_test):
    resultados = []
    X_normal = X_train[y_train == 0]
    for n_est, cont in itertools.product([100, 200, 300], [0.005, 0.01, 0.02, 0.05]):
        t0 = time.perf_counter()
        model = IsolationForest(
            n_estimators=n_est, contamination=cont, random_state=config.SEED,
        ).fit(X_normal)
        t_fit = time.perf_counter() - t0
        raw = model.predict(X_test)
        y_pred = np.where(raw == -1, 1, 0)
        m = _evaluar(y_test, y_pred)
        m.update({"algoritmo": "IsolationForest", "n_estimators": n_est,
                  "contamination": cont, "tiempo_fit_s": round(t_fit, 4)})
        resultados.append(m)
    return resultados


def grid_one_class_svm(X_train, y_train, X_test, y_test):
    resultados = []
    X_normal = X_train[y_train == 0]
    for nu, gamma in itertools.product([0.005, 0.01, 0.02, 0.03, 0.05], ["scale", "auto"]):
        t0 = time.perf_counter()
        model = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu).fit(X_normal)
        t_fit = time.perf_counter() - t0
        raw = model.predict(X_test)
        y_pred = np.where(raw == -1, 1, 0)
        m = _evaluar(y_test, y_pred)
        m.update({"algoritmo": "OneClassSVM", "nu": nu, "gamma": gamma,
                  "tiempo_fit_s": round(t_fit, 4)})
        resultados.append(m)
    return resultados


def grid_autoencoder(X_train, y_train, X_test, y_test):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("[SKIP] Autoencoder: torch no instalado.")
        return []

    torch.manual_seed(config.SEED)
    X_normal = X_train[y_train == 0].astype(np.float32)
    X_test_f = X_test.astype(np.float32)
    n_features = X_normal.shape[1]

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

    resultados = []
    for epochs, percentil in itertools.product([50, 80, 120], [95, 97, 99]):
        model = AE()
        optim = torch.optim.Adam(model.parameters(), lr=config.AE_LR)
        loss_fn = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_normal)),
            batch_size=config.AE_BATCH_SIZE, shuffle=True,
        )
        t0 = time.perf_counter()
        model.train()
        for _ in range(epochs):
            for (batch,) in loader:
                optim.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optim.step()
        t_fit = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            err_train = ((model(torch.from_numpy(X_normal)) - torch.from_numpy(X_normal)) ** 2).mean(dim=1).numpy()
            umbral = float(np.percentile(err_train, percentil))
            err_test = ((model(torch.from_numpy(X_test_f)) - torch.from_numpy(X_test_f)) ** 2).mean(dim=1).numpy()
            y_pred = (err_test > umbral).astype(int)
        m = _evaluar(y_test, y_pred)
        m.update({"algoritmo": "Autoencoder", "epochs": epochs,
                  "percentil": percentil, "tiempo_fit_s": round(t_fit, 4)})
        resultados.append(m)
    return resultados


def main():
    print("[grid] Cargando datos...")
    X_train, X_test, y_train, y_test, features = _cargar_datos()
    print(f"[grid] Train: {len(X_train)} | Test: {len(X_test)}")

    todos = []
    for nombre, fn in [
        ("IsolationForest", grid_isolation_forest),
        ("OneClassSVM", grid_one_class_svm),
        ("Autoencoder", grid_autoencoder),
    ]:
        print(f"[grid] {nombre}...")
        todos.extend(fn(X_train, y_train, X_test, y_test))

    if not todos:
        print("[grid] Sin resultados.")
        return

    df = pd.DataFrame(todos).sort_values(["algoritmo", "f1"], ascending=[True, False])
    ruta = RESULTS_DIR / "grid_search_resultados.csv"
    df.to_csv(ruta, index=False)

    print(f"\n[grid] Resultados en {ruta}\n")
    for alg in df["algoritmo"].unique():
        sub = df[df["algoritmo"] == alg].head(3)
        print(f"\n=== {alg} (top 3 por F1) ===")
        cols = [c for c in ["n_estimators", "contamination", "nu", "gamma",
                            "epochs", "percentil", "f1", "precision",
                            "recall", "fpr", "tiempo_fit_s"] if c in sub.columns]
        print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()