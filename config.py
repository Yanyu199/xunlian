import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODEXDATA_DIR = os.path.abspath(os.environ.get("TEM_CODEXDATA_DIR", os.path.join(BASE_DIR, os.pardir)))
OUTPUT_DIR = os.path.abspath(os.environ.get("TEM_OUTPUT_DIR", os.path.join(BASE_DIR, "output")))
DATA_DIR = os.path.abspath(os.environ.get("TEM_DATA_DIR", os.path.join(BASE_DIR, "data")))
TRAINING_JOBS_DIR = os.path.abspath(os.environ.get("TEM_TRAINING_JOBS_DIR", os.path.join(CODEXDATA_DIR, "training_jobs")))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAINING_JOBS_DIR, exist_ok=True)

# Forward modelling and sample settings
TIME_CHANNELS = 30
TIME_MIN = 1e-5
TIME_MAX = 1e-2
LAYER_NUM = 5
R_MIN, R_MAX = 10, 1000
THICKNESS_MIN, THICKNESS_MAX = 10, 100
SAMPLE_SIZE = 50
RANDOM_SEED = 42

# Training settings
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TRAIN_PORTION = 0.8
VALID_PORTION = 0.2
DEVICE = os.environ.get("TEM_DEVICE", "cpu")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_tem_model.pt")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "data_scaler.json")
TRAIN_HISTORY_PATH = os.path.join(OUTPUT_DIR, "train_history.json")
ROOT_MODEL_SAVE_TEMPLATE = os.path.join(CODEXDATA_DIR, "tem_model_{job_id}.pt")
ROOT_SCALER_SAVE_TEMPLATE = os.path.join(CODEXDATA_DIR, "tem_model_{job_id}_scaler.json")
ROOT_HISTORY_SAVE_TEMPLATE = os.path.join(CODEXDATA_DIR, "tem_model_{job_id}_history.json")

# API settings
API_TITLE = "TEM Borehole Inversion Backend"
