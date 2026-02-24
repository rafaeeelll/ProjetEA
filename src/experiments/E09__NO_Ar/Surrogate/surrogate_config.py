from pathlib import Path


E09_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]

OUTPUT_DIR = PROJECT_ROOT.joinpath("outputs", "thrust_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DATASET_PATH = OUTPUT_DIR.joinpath("thrust_dataset_msis.json")
GP_DATASET_PATH = OUTPUT_DIR.joinpath("thrust_gp_dataset.json")
GP_MODEL_PATH = OUTPUT_DIR.joinpath("thrust_gp_model.pkl")
GP_MODEL_META_PATH = OUTPUT_DIR.joinpath("thrust_gp_model.json")
BO_LOG_PATH = OUTPUT_DIR.joinpath("bo_maxmin_ei_log.json")

# Feature engineering
ARGON_LOG_EPS = 1e14
MDOT_LOG_EPS = 1e-20
ETA_COLLECTION = 0.35
MAX_TRAIN_SAMPLES = 1500
TRAIN_SUBSAMPLE_SEED = 42

# Physics simplification for optimization
CD_FRONT = 2.2
