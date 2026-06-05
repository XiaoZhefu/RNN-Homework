from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
STEPS = [
    ("1. 数据准备", "step1_preparing.py"),
    ("2. 模型选择与结构设计", "step2_model_design.py"),
    ("3. 模型训练", "step3_training.py"),
    ("4. 模型评估", "step4_evaluation.py"),
]


def main():
    print("=" * 60)
    print("  RNN Homework 总控脚本")
    print("=" * 60)

    for title, script_name in STEPS:
        print("-" * 60)
        print(f"  [Step] {title}")
        print(f"  [File] {script_name}")
        print("-" * 60)

        result = subprocess.run([sys.executable, "-B", script_name], cwd=SCRIPT_DIR)
        if result.returncode != 0:
            print("-" * 60)
            print(f"  [Error] {script_name} 运行失败，退出码: {result.returncode}")
            print("=" * 60)
            raise SystemExit(result.returncode)

    print("-" * 60)
    print("  [Done] 所有 Python 步骤运行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
