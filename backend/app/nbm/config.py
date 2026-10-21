"""Constantes del modulo NBM."""
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASET_CSV = RESULTS_DIR / "dataset_sintetico.csv"
COMPARATIVA_CSV = RESULTS_DIR / "comparativa_algoritmos.csv"

# Dataset sintetico
SEED = 42
N_REGISTROS = 2000
PROPORCION_ANOMALIAS = 0.05  # 5%
TEST_SIZE = 0.30

# Isolation Forest
IF_N_ESTIMATORS = 100
IF_CONTAMINATION = PROPORCION_ANOMALIAS

# One-Class SVM
OCSVM_KERNEL = "rbf"
OCSVM_GAMMA = "scale"
OCSVM_NU = PROPORCION_ANOMALIAS

# Autoencoder
AE_EPOCHS = 50
AE_BATCH_SIZE = 64
AE_LR = 1e-3
AE_HIDDEN = 8  # dimension del cuello de botella
AE_UMBRAL_PERCENTIL = 95  # anomalia si error de reconstruccion > p95 del train normal