# run_pipeline.py
import subprocess
import sys
import os
import time


def run_script(script_name):
    print(f"\n" + "=" * 60)
    print(f"🚀 [正在执行阶段]: {script_name}")
    print("=" * 60 + "\n")

    # 使用 subprocess 运行，确保每个脚本在独立的进程中运行
    # 这样可以彻底释放 CuPy 和 PyTorch 占用的 GPU 显存，防止 OOM
    start_time = time.time()
    result = subprocess.run([sys.executable, script_name])
    cost_time = time.time() - start_time

    if result.returncode != 0:
        print(f"\n❌ [致命错误]: {script_name} 运行失败！流水线已紧急终止。")
        sys.exit(1)
    else:
        print(f"\n✅ [阶段完成]: {script_name} 运行成功！(耗时: {cost_time:.2f} 秒)")


if __name__ == "__main__":
    print("\n🌟 欢迎使用 TEM 离线训练全自动流水线 🌟")
    print("请确保你已经将真实的 dBzdt.txt 放入了 data/ 目录中。\n")

    # 确保必要的文件夹存在
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # 定义要按顺序执行的脚本列表
    pipeline_scripts = [
        "01_prior_analyzer.py",
        "02_data_generator.py",
        "03_train.py"
    ]

    total_start = time.time()

    # 依次执行
    for script in pipeline_scripts:
        if not os.path.exists(script):
            print(f"❌ 找不到文件: {script}，请检查目录结构！")
            sys.exit(1)
        run_script(script)

    total_cost = time.time() - total_start
    print("\n" + "★" * 60)
    print(f"🎉 恭喜！全部流程执行完毕！总耗时: {total_cost:.2f} 秒")
    print("📦 你的最优模型 (best_tem_model.pt) 和 归一化参数 (data_scaler.json)")
    print("📦 已经安全保存在 output/ 目录下，随时可以提供给后端使用了！")
    print("★" * 60 + "\n")