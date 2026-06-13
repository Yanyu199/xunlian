# tem_model_factory/01_prior_analyzer.py
import numpy as np
import json
import os
from config import DATA_DIR, OUTPUT_DIR

# 瞬变电磁物理常数
MU_0 = 4 * np.pi * 1e-7


def parse_dbzdt_txt(file_path):
    """
    解析你的 dBzdt.txt 文件
    你的文件格式：第一行是表头(time[s]  1.00  2.00 ...)，后面第一列是时间，后面列是测点响应值
    """
    times = []
    dbzdt_data = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:  # 跳过第一行表头
        parts = line.strip().split()
        if len(parts) >= 2:
            times.append(float(parts[0]))
            # 取第一条测线的数据作为参考 (或者取所有测线的平均值)
            # 注意：取绝对值，因为早晚期可能存在符号变化
            dbzdt_data.append(abs(float(parts[1])))

    return np.array(times), np.array(dbzdt_data)


def calculate_apparent_resistivity(t, dbzdt, tx_area=4.0, rx_area=1.0, current=1.0):
    """
    利用瞬变电磁中心回线晚期视电阻率公式估算电阻率
    公式: rho(t) = (mu_0 / (4*pi*t)) * (2 * mu_0 * M / (5 * t * V(t)))^(2/3)
    这里 M 为磁矩 (tx_area * current)
    """
    M = tx_area * current
    # 避免除以 0
    dbzdt = np.where(dbzdt == 0, 1e-15, dbzdt)

    term1 = MU_0 / (4 * np.pi * t)
    term2 = (2 * MU_0 * M) / (5 * t * dbzdt * rx_area)

    rho_app = term1 * (term2 ** (2 / 3))
    return rho_app


def analyze_prior_info():
    print("=== 开始运行 01: 先验信息提取 (Prior Analyzer) ===")

    txt_path = os.path.join(DATA_DIR, "dBzdt.txt")
    if not os.path.exists(txt_path):
        print(f"❌ 未找到野外数据文件: {txt_path}")
        print("💡 请将 dBzdt.txt 放入 data/ 文件夹中。")
        return

    # 1. 解析数据
    times, dbzdt = parse_dbzdt_txt(txt_path)
    print(f"✅ 成功读取 dBzdt.txt，共包含 {len(times)} 个时间道。")

    # 2. 计算晚期视电阻率曲线
    # 这里假设发射线框为 2m x 2m = 4.0 (依据你代码中 tx_size_key='4')
    rho_app = calculate_apparent_resistivity(times, dbzdt, tx_area=4.0)

    # 3. 统计范围 (去除可能出现的极端异常值，取 5% ~ 95% 分位数)
    # 因为早期数据可能不满足晚期公式条件，我们取中晚期数据估算
    valid_rho = rho_app[len(rho_app) // 3:]

    r_min_est = np.percentile(valid_rho, 5)
    r_max_est = np.percentile(valid_rho, 95)

    # 增加一个物理宽容度 (向下扩展半个数量级，向上扩展一个数量级)
    r_min_final = max(1.0, r_min_est * 0.5)
    r_max_final = r_max_est * 5.0

    print("\n📊 视电阻率估算结果:")
    print(f"   估计最小电阻率: {r_min_est:.2f} Ω·m")
    print(f"   估计最大电阻率: {r_max_est:.2f} Ω·m")
    print(f"🎯 最终建议样本搜索范围: [{r_min_final:.1f}, {r_max_final:.1f}] Ω·m")

    # 4. 保存先验信息供下一步生成样本使用
    prior_config = {
        "r_min": float(r_min_final),
        "r_max": float(r_max_final),
        "time_channels": len(times),
        "t_min": float(times[0]),
        "t_max": float(times[-1])
    }

    prior_path = os.path.join(OUTPUT_DIR, "prior_bounds.json")
    with open(prior_path, "w") as f:
        json.dump(prior_config, f, indent=4)

    print(f"✅ 先验参数已保存至: {prior_path}")
    print("🚀 现在你可以运行 02_data_generator.py 了！")


if __name__ == "__main__":
    analyze_prior_info()