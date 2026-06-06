from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PROJECT_DIR / "scripts"
MODEL_DIR = PROJECT_DIR / "models"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
STEPS = [
    ("1. 数据准备", "step1_preparing.py"),
    ("2. 模型选择与结构设计", "step2_model_design.py"),
    ("3. 模型训练", "step3_training.py"),
    ("4. 迁移学习", "step4_transfer_learning.py"),
    ("5. 模型评估", "step5_evaluation.py"),
]
STEP3_WEIGHTS = [
    MODEL_DIR / "lstm_best.pt",
    MODEL_DIR / "gru_best.pt",
    MODEL_DIR / "no_smoothing_lstm_best.pt",
    MODEL_DIR / "no_smoothing_gru_best.pt",
]
TRANSFER_WEIGHTS = [
    MODEL_DIR / "transfer_lstm_best.pt",
    MODEL_DIR / "transfer_gru_best.pt",
]
STEP1_OUTPUT_FILES = [
    PROCESSED_DATA_DIR / "cleaned_dataset.csv",
    PROCESSED_DATA_DIR / "cleaned_dataset_no_smoothing.csv",
]


def all_exist(paths):
    return all(path.exists() for path in paths)


def select_steps():
    has_step3_weights = all_exist(STEP3_WEIGHTS)
    has_transfer_weights = all_exist(TRANSFER_WEIGHTS)
    has_step1_outputs = all_exist(STEP1_OUTPUT_FILES)

    if has_step3_weights and has_transfer_weights:
        print("  [Skip] 已检测到训练权重与迁移学习权重，直接进入 Step 5")
        return STEPS[4:]

    if has_step3_weights:
        print("  [Skip] 已检测到训练权重，未检测到迁移学习权重，直接进入 Step 4")
        return STEPS[3:]

    if has_step1_outputs:
        print("  [Skip] 已检测到清洗后数据，直接进入 Step 3")
        return STEPS[2:]

    print("  [Start] 未检测到完整训练权重，从 Step 1 开始运行")
    return STEPS


def main():
    print("=" * 60)
    print("  RNN Homework 总控脚本")
    print("=" * 60)

    for title, script_name in select_steps():
        print("-" * 60)
        print(f"  [Step] {title}")
        print(f"  [File] scripts/{script_name}")
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
