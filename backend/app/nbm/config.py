"""Constantes del modulo NBM."""
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASET_CSV = RESULTS_DIR / "dataset_sintetico.csv"
COMPARATIVA_CSV = RESULTS_DIR / "comparativa_algoritmos.csv"

# Directorio de modelos entrenados
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "oneclasssvm_v1.joblib"
# Dataset sintetico
SEED = 42
N_REGISTROS = 50000          # antes: 2000
PROPORCION_ANOMALIAS = 0.01  # antes: 0.05 (1% es mas realista)
TEST_SIZE = 0.20             # antes: 0.30 (20% = 10.000 eventos en test)

# Isolation Forest (mejor de grid search)
IF_N_ESTIMATORS = 100
IF_CONTAMINATION = 0.01

# OneClassSVM (mejor de grid search)
OCSVM_KERNEL = "rbf"
OCSVM_GAMMA = "auto"
OCSVM_NU = 0.005

# Autoencoder (mejor de grid search)
AE_EPOCHS = 80              # 120 da F1=0.7328 pero tarda 362s; 80 da F1=0.7273 y tarda 249s
AE_BATCH_SIZE = 128
AE_LR = 1e-3
AE_HIDDEN = 8
AE_UMBRAL_PERCENTIL = 99