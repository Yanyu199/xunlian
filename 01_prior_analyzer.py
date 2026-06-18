# tem_model_factory/01_prior_analyzer.py
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, squareform
from core.forward_tem import TEMForwardModeler
from config import DATA_DIR, OUTPUT_DIR, LAYER_NUM, TIME_CHANNELS


def parse_dbzdt_txt(file_path):
    """
    读取真实野外数据，返回形状为 (N_samples, T_channels) 的二维数组
    假设第一列是时间，后续每一列是一个测点的响应数据
    """
    times = []
    dbzdt_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:  # 跳过表头
        parts = line.strip().split()
        if len(parts) >= 2:
            times.append(float(parts[0]))
            # 后续所有列为不同测点的数据
            dbzdt_data.append([abs(float(x)) for x in parts[1:]])

    times = np.array(times)
    # 转置，使行代表测点 (Samples)，列代表时间道 (Time Channels)
    real_data = np.array(dbzdt_data).T
    return times, real_data


def extract_features(data, t_vec):
    """
    提取晚期均值、对数斜率、最大幅值、时间窗口能量等特征 (对齐MATLAB逻辑)
    data shape: (N, T)
    """
    N, T = data.shape
    n_windows = 5
    edges = np.linspace(0, T, n_windows + 1, dtype=int)
    feats = np.zeros((N, n_windows + 3))

    for i in range(N):
        tr = data[i, :]
        wE = []
        # 窗口能量
        for w in range(n_windows):
            s, e = edges[w], edges[w + 1]
            seg = tr[s:e] if e > s else np.array([tr[s]])
            wE.append(np.sqrt(np.sum(seg ** 2)))

        y = np.abs(tr) + 1e-12
        start_idx = max(1, int(0.02 * T))

        # 晚期衰减斜率
        try:
            slope, _ = np.polyfit(np.log(t_vec[start_idx:]), np.log(y[start_idx:]), 1)
        except:
            slope = 0.0

        late_mean = np.mean(y[int(0.7 * T):])
        mx = np.max(y)

        feats[i, :] = wE + [slope, mx, late_mean]
    return feats


def compute_mmd(X, Y):
    """计算最大均值差异(MMD)，使用中位数启发式选择高斯核sigma"""
    Z = np.vstack((X, Y))
    dists = pdist(Z, 'euclidean')
    sigma = np.median(dists) if len(dists) > 0 else 1.0
    sigma = max(sigma, 1e-6)

    gamma = 1.0 / (2 * sigma ** 2)
    XX = np.exp(-gamma * squareform(pdist(X, 'sqeuclidean')))
    YY = np.exp(-gamma * squareform(pdist(Y, 'sqeuclidean')))
    XY = np.exp(-gamma * squareform(pdist(np.vstack((X, Y)), 'sqeuclidean'))[:len(X), len(X):])

    n, m = len(X), len(Y)
    sumXX = (np.sum(XX) - np.trace(XX)) / max(n * (n - 1), 1)
    sumYY = (np.sum(YY) - np.trace(YY)) / max(m * (m - 1), 1)
    sumXY = np.sum(XY) / max(n * m, 1)

    mmd_val = max(sumXX + sumYY - 2 * sumXY, 0)
    return mmd_val


def classifier_auc(X_sim, X_real):
    """训练随机森林分辨真实和模拟数据，返回AUC (越接近0.5越好)"""
    X = np.vstack((X_sim, X_real))
    y = np.hstack((np.zeros(len(X_sim)), np.ones(len(X_real))))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.zeros(len(y))

    for train_idx, test_idx in kf.split(X):
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        probs[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    auc = roc_auc_score(y, probs)
    return auc if auc >= 0.5 else 1.0 - auc


def target_func(r_min, r_max, times, feat_real, modeler, n_sim=100):
    """贝叶斯优化的目标函数：正演生成数据 -> 提取特征 -> 计算综合得分"""
    if r_max <= r_min:
        return -1e6

    # 对数空间均匀采样地层参数
    log_r_min, log_r_max = np.log10(r_min), np.log10(r_max)
    rhos = 10 ** np.random.uniform(log_r_min, log_r_max, (n_sim, LAYER_NUM))
    thicknesses = 10 ** np.random.uniform(np.log10(10), np.log10(100), (n_sim, LAYER_NUM - 1))

    # 批量正演生成模拟数据
    sim_data = modeler.forward_batch(rhos, thicknesses, times)
    feat_sim = extract_features(sim_data, times)

    # Z-score 标准化
    all_stack = np.vstack((feat_real, feat_sim))
    mean, std = np.mean(all_stack, axis=0), np.std(all_stack, axis=0) + 1e-9
    X_real = (feat_real - mean) / std
    X_sim = (feat_sim - mean) / std

    mmd_val = compute_mmd(X_sim, X_real)
    auc_val = classifier_auc(X_sim, X_real)

    # 综合得分：MMD越小越好，AUC越接近0.5越好
    score = mmd_val + 0.2 * abs(auc_val - 0.5)
    return -score, mmd_val, auc_val


def analyze_prior_info():
    print("=== 开始运行 01: 先验信息提取 (Bayesian Prior Analyzer) ===")
    txt_path = os.path.join(DATA_DIR, "dBzdt.txt")

    if not os.path.exists(txt_path):
        print(f"❌ 未找到野外数据文件: {txt_path}")
        return

    times, real_data = parse_dbzdt_txt(txt_path)
    feat_real = extract_features(real_data, times)
    print(f"✅ 成功读取真实数据，提取特征维度: {feat_real.shape}")

    modeler = TEMForwardModeler()  # 初始化GPU正演引擎

    # 贝叶斯优化参数
    init_points = 30
    n_iter = 15
    r_min_bounds = [1.0, 50.0]
    r_max_bounds = [10.0, 300.0]

    all_params, all_scores = [], []

    print("\n--- 阶段 1: 随机探索 ---")
    for i in range(init_points):
        r_min = np.random.uniform(r_min_bounds[0], r_min_bounds[1])
        r_max = np.random.uniform(max(r_min + 10, r_max_bounds[0]), r_max_bounds[1])
        score, mmd, auc = target_func(r_min, r_max, times, feat_real, modeler)
        all_params.append([r_min, r_max])
        all_scores.append(score)
        print(
            f"[随机 {i + 1}/{init_points}] r_min={r_min:.1f}, r_max={r_max:.1f} | MMD={mmd:.4f}, AUC={auc:.4f}, Score={-score:.4f}")

    print("\n--- 阶段 2: 局部寻优 ---")
    for i in range(n_iter):
        best_idx = np.argmax(all_scores)
        best_r_min, best_r_max = all_params[best_idx]

        # 围绕最优解加入高斯噪声探索
        r_min = np.clip(best_r_min + np.random.normal(0, 5), r_min_bounds[0], r_min_bounds[1])
        r_max = np.clip(best_r_max + np.random.normal(0, 20), r_min + 10, r_max_bounds[1])

        score, mmd, auc = target_func(r_min, r_max, times, feat_real, modeler)
        all_params.append([r_min, r_max])
        all_scores.append(score)
        print(f"[寻优 {i + 1}/{n_iter}] r_min={r_min:.1f}, r_max={r_max:.1f} | Score={-score:.4f}")

    best_idx = np.argmax(all_scores)
    final_r_min, final_r_max = all_params[best_idx]

    print(f"\n🎯 最终建议样本搜索范围: [{final_r_min:.1f}, {final_r_max:.1f}] Ω·m")

    prior_config = {
        "r_min": float(final_r_min), "r_max": float(final_r_max),
        "time_channels": len(times)
    }
    with open(os.path.join(OUTPUT_DIR, "prior_bounds.json"), "w") as f:
        json.dump(prior_config, f, indent=4)
    print("🚀 先验分布对齐完成，可以运行 02_data_generator.py 了！")


if __name__ == "__main__":
    analyze_prior_info()