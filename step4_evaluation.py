from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from step2_model_design import (
    BATCH_SIZE,
    GRURegressor,
    LSTMRegressor,
    TARGET_COLUMNS,
    print_tag,
)
from step3_training import (
    DATASET_CONFIGS,
    MODEL_DIR,
    OUTPUT_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    load_sequence_data,
    split_train_val_test,
)


# 评估结果、预测结果和 MATLAB 可视化脚本路径
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PIC_DIR = SCRIPT_DIR / "outputpics"
MATLAB_SCRIPT = SCRIPT_DIR / "matlab_visualization_evaluation.m"
# 两套预处理方案的评估输出；原有方案保持原文件名，新增方案使用后缀
EVALUATION_CONFIGS = [
    {
        "name": "平滑+归一化",
        "data_path": DATASET_CONFIGS[0]["data_path"],
        "model_prefix": DATASET_CONFIGS[0]["model_prefix"],
        "metrics_path": OUTPUT_DIR / "evaluation_metrics.csv",
        "prediction_path": OUTPUT_DIR / "prediction_results.csv",
    },
    {
        "name": "仅归一化",
        "data_path": DATASET_CONFIGS[1]["data_path"],
        "model_prefix": DATASET_CONFIGS[1]["model_prefix"],
        "metrics_path": OUTPUT_DIR / "evaluation_metrics_no_smoothing.csv",
        "prediction_path": OUTPUT_DIR / "prediction_results_no_smoothing.csv",
    },
]


def build_test_loader(x_test, y_test):
    """构造测试集 DataLoader，测试阶段不打乱样本顺序。"""
    dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def load_model(model_name, model_prefix, device):
    """按模型名称加载训练阶段保存的最优模型权重。"""
    if model_name == "LSTM":
        model = LSTMRegressor()
    elif model_name == "GRU":
        model = GRURegressor()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model_path = MODEL_DIR / f"{model_prefix}{model_name.lower()}_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, dataloader, device):
    """在测试集上进行预测，并返回预测值与真实值。"""
    predictions = []
    truths = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            pred_y = model(batch_x).cpu().numpy()
            predictions.append(pred_y)
            truths.append(batch_y.numpy())
    return np.vstack(predictions), np.vstack(truths)


def calculate_metrics(model_name, y_true, y_pred):
    """计算每个目标以及整体的 MSE、RMSE、MAE。"""
    rows = []
    for index, target in enumerate(TARGET_COLUMNS):
        error = y_pred[:, index] - y_true[:, index]
        mse = float(np.mean(error ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(error)))
        rows.append({"model": model_name, "target": target, "mse": mse, "rmse": rmse, "mae": mae})

    error = y_pred - y_true
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    rows.append({"model": model_name, "target": "Overall", "mse": mse, "rmse": rmse, "mae": mae})
    return rows


def save_prediction_results(prediction_path, sample_index, y_true, model_predictions):
    """保存测试集真实值和各模型预测值，供 MATLAB 统一绘图。"""
    result = pd.DataFrame({"sample_index": sample_index})
    for index, target in enumerate(TARGET_COLUMNS):
        result[f"{target}_true"] = y_true[:, index]
        for model_name, y_pred in model_predictions.items():
            result[f"{model_name}_{target}_pred"] = y_pred[:, index]
    result.to_csv(prediction_path, index=False, encoding="utf-8-sig")


def evaluate_dataset(config, device):
    """评估某一种预处理数据下的 LSTM 和 GRU 模型。"""
    print("-" * 60)
    print_tag("Data", f"当前预处理方案: {config['name']}")
    print_tag("File", f"测试数据: {config['data_path'].name}")

    x_seq, y_seq = load_sequence_data(config["data_path"])
    _, _, _, _, x_test, y_test = split_train_val_test(x_seq, y_seq, TRAIN_RATIO, VAL_RATIO)
    test_loader = build_test_loader(x_test, y_test)

    print_tag("Input", f"测试集 X: {x_test.shape}")
    print_tag("Label", f"测试集 y: {y_test.shape}")
    print_tag("Device", str(device))
    print("-" * 60)

    all_metrics = []
    model_predictions = {}
    y_true_ref = None
    for model_name in ["LSTM", "GRU"]:
        model = load_model(model_name, config["model_prefix"], device)
        y_pred, y_true = predict(model, test_loader, device)
        model_predictions[model_name] = y_pred
        y_true_ref = y_true
        all_metrics.extend(calculate_metrics(model_name, y_true, y_pred))

        overall = [row for row in all_metrics if row["model"] == model_name and row["target"] == "Overall"][0]
        print_tag("Metric", f"{model_name} Overall RMSE: {overall['rmse']:.6f}, MAE: {overall['mae']:.6f}")

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(config["metrics_path"], index=False, encoding="utf-8-sig")
    sample_index = np.arange(len(y_true_ref))
    save_prediction_results(config["prediction_path"], sample_index, y_true_ref, model_predictions)

    print("-" * 60)
    print_tag("File", f"评价指标: {config['metrics_path']}")
    print_tag("File", f"预测结果: {config['prediction_path']}")


def run_matlab_visualization():
    """调用独立 MATLAB 脚本绘制损失曲线、指标图和预测对比图。"""
    matlab_exe = shutil.which("matlab")
    if matlab_exe is None:
        print_tag("Skip", "未找到 MATLAB，跳过可视化绘图")
        return
    subprocess.run([matlab_exe, "-batch", f"run('{MATLAB_SCRIPT.as_posix()}')"], cwd=SCRIPT_DIR, check=True)


def main():
    print("=" * 60)
    print("  4. 模型评估")
    print("=" * 60)
    print("  [Step] 在测试集上计算 MSE、RMSE、MAE")
    print("  [File] 模型权重 : models/")
    print("-" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_PIC_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for config in EVALUATION_CONFIGS:
        evaluate_dataset(config, device)

    print("-" * 60)
    print("  [Step] 调用 MATLAB 绘制损失曲线、指标图与预测对比图")
    print("  [File] MATLAB脚本 : matlab_visualization_evaluation.m")
    run_matlab_visualization()
    print("-" * 60)
    print("  [Done] 模型评估完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
