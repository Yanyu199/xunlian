import numpy as np
import multiprocessing
from config import *
import json
import os

# 假设你在 core.forward_tem 中封装了正演计算函数 compute_tem_response
# from core.forward_tem import compute_tem_response
def generate_dataset_gpu():
    print(f"开始生成 {SAMPLE_SIZE} 个样本数据...")

    # [新增部分] 自动读取 01 步骤估算的先验电阻率范围
    prior_path = os.path.join(OUTPUT_DIR, "prior_bounds.json")
    if os.path.exists(prior_path):
        with open(prior_path, "r") as f:
            prior = json.load(f)
            r_min = prior["r_min"]
            r_max = prior["r_max"]
            print(f"✅ 成功加载先验范围: {r_min:.1f} - {r_max:.1f} Ω·m")
    else:
        # 如果没有运行01，使用 config.py 的默认值
        r_min, r_max = R_MIN, R_MAX
        print(f"⚠️ 未找到 prior_bounds.json，使用默认范围: {r_min} - {r_max} Ω·m")

    # 2. 预先随机生成所有地电模型参数 (使用刚刚读到的 r_min, r_max)
    all_rhos = np.random.uniform(r_min, r_max, (SAMPLE_SIZE, LAYER_NUM))
def generate_single_sample(seed):
    np.random.seed(seed)
    # 1. 随机生成地层厚度和电阻率 (使用 config 中的 R_MIN, R_MAX)
    resistivities = np.random.uniform(R_MIN, R_MAX, LAYER_NUM)
    thicknesses = np.random.uniform(10, 100, LAYER_NUM - 1)

    # 2. 调用原有的正演物理算法计算响应
    # dbzdt = compute_tem_response(resistivities, thicknesses)
    dbzdt = np.random.rand(TIME_CHANNELS) * 1e-5  # 模拟生成的假数据

    return dbzdt, resistivities, thicknesses


if __name__ == "__main__":
    print(f"开始生成 {SAMPLE_SIZE} 个样本数据，范围: {R_MIN} - {R_MAX} Ω·m")
    # 使用多进程加速你的正演生成（你的原代码有并行逻辑，这里做优雅封装）
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(generate_single_sample, range(SAMPLE_SIZE))

    # 将结果保存为 numpy 数组或你原本 dataset.py 需要的 .txt 格式
    X_data = np.array([res[0] for res in results])
    Y_res = np.array([res[1] for res in results])
    Y_thick = np.array([res[2] for res in results])

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_data)
    np.save(os.path.join(DATA_DIR, "Y_train.npy"), np.hstack((Y_res, Y_thick)))
    print("样本生成完毕！")