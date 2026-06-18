# tem_model_factory/02_data_generator.py
import numpy as np
import json
import os
import time
from config import *
from core.forward_tem import TEMForwardModeler


def main():
    print("=== 开始运行 02: 瞬变电磁正演-真实物理样本生成 ===")

    # 1. 尝试读取 01 步骤中贝叶斯优化输出的精确范围
    prior_path = os.path.join(OUTPUT_DIR, "prior_bounds.json")
    r_min, r_max = R_MIN, R_MAX
    if os.path.exists(prior_path):
        with open(prior_path, "r") as f:
            prior = json.load(f)
            r_min = prior["r_min"]
            r_max = prior["r_max"]
            print(f"✅ 成功加载先验范围: [{r_min:.1f}, {r_max:.1f}] Ω·m")
    else:
        print(f"⚠️ 未找到 prior_bounds.json，使用 config.py 默认范围: [{r_min}, {r_max}] Ω·m")

    # 2. 生成符合物理约束的地层参数 (在 CPU 上准备数据)
    print(f"🚀 正在生成 {SAMPLE_SIZE} 个地电模型参数 (对数空间采样 + 层厚约束)...")

    # 2.1 对数空间生成电阻率
    log_r_min, log_r_max = np.log10(r_min), np.log10(r_max)
    rhos = 10 ** np.random.uniform(log_r_min, log_r_max, (SAMPLE_SIZE, LAYER_NUM))

    # 2.2 约束生成地层厚度 (必须保证相邻层差异 >= 0.99，保证地层可分辨)
    thicknesses = np.zeros((SAMPLE_SIZE, LAYER_NUM - 1))
    for i in range(SAMPLE_SIZE):
        while True:
            th_vals = 10 ** np.random.uniform(np.log10(10), np.log10(100), LAYER_NUM - 1)
            if LAYER_NUM > 1:
                differ = np.abs(th_vals[1:] - th_vals[:-1]) / (th_vals[:-1] + 1e-12)
                if np.all(differ >= 0.99):
                    thicknesses[i] = th_vals
                    break
            else:
                thicknesses[i] = th_vals
                break

    # 3. 初始化 GPU 正演物理引擎
    print("⏳ 正在初始化 GPU 正演物理引擎...")
    modeler = TEMForwardModeler(tx_size_key='4', center_z=0.0)

    # 定义观测时间道 (对齐 MATLAB 及实际仪器的晚期时间窗口，如 1e-5 到 1e-2 秒)
    time_gates = np.logspace(-5, -2, TIME_CHANNELS)

    # 4. GPU 批量计算响应
    # 注意：防止 GPU 显存溢出 (OOM)，将十万量级的样本分批送入 GPU
    GPU_BATCH_SIZE = 50  # 可根据你的显卡显存(VRAM)大小适当调大(如128)或调小(如10)
    all_dbzdt = []

    total_batches = (SAMPLE_SIZE + GPU_BATCH_SIZE - 1) // GPU_BATCH_SIZE
    print(f"🔥 开始进行麦克斯韦方程数值求解 (分 {total_batches} 批次在 GPU 上计算)...")

    start_time = time.time()

    for b_idx in range(total_batches):
        st = b_idx * GPU_BATCH_SIZE
        ed = min(st + GPU_BATCH_SIZE, SAMPLE_SIZE)

        # 切片获取当前 Batch 的参数
        rho_batch = rhos[st:ed]
        thick_batch = thicknesses[st:ed]

        # 调用物理模型 (结果返回到 CPU)
        dbzdt_batch = modeler.forward_batch(rho_batch, thick_batch, time_gates)
        all_dbzdt.append(dbzdt_batch)

        if (b_idx + 1) % max(1, total_batches // 10) == 0 or (b_idx + 1) == total_batches:
            progress = (b_idx + 1) / total_batches * 100
            print(f"   -> 正演计算进度: {progress:.1f}% ({st}/{SAMPLE_SIZE})")

    # 5. 合并并保存结果
    X_data = np.concatenate(all_dbzdt, axis=0)  # (SAMPLE_SIZE, TIME_CHANNELS)
    Y_data = np.hstack((rhos, thicknesses))  # (SAMPLE_SIZE, LAYER_NUM * 2 - 1)

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_data)
    np.save(os.path.join(DATA_DIR, "Y_train.npy"), Y_data)

    elapsed_time = time.time() - start_time
    print("\n✅ 真实物理样本生成完毕！")
    print(f"   总耗时: {elapsed_time:.2f} 秒")
    print(f"   X_train 维度: {X_data.shape} (真实物理衰减信号 dBz/dt)")
    print(f"   Y_train 维度: {Y_data.shape} (地层电阻率与厚度)")
    print(f"   数据已落盘至: {DATA_DIR}/")
    print("🚀 现在你可以直接运行 03_train.py 进行网络训练了！")


if __name__ == "__main__":
    main()