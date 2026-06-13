import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 1. 先验与正演参数
TIME_CHANNELS = 30           # 瞬变电磁时间道数
LAYER_NUM = 5                # 反演地层层数
R_MIN, R_MAX = 10, 1000      # 默认电阻率范围（会被步骤1更新）
SAMPLE_SIZE = 50000          # 需要生成的样本总数

# 2. 深度学习训练参数
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DEVICE = "cuda" # 或者 "cpu"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_tem_model.pt")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "data_scaler.json")