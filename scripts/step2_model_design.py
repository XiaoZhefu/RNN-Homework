from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# 基础路径与数据文件
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
DATA_PATH = PROCESSED_DATA_DIR / "cleaned_dataset.csv"
NO_SMOOTHING_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_dataset_no_smoothing.csv"
# 时间序列样本与模型超参数
SEQ_LEN = 50
BATCH_SIZE = 32
INPUT_SIZE = 7
OUTPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
PRINT_TAG_WIDTH = 9


def print_tag(tag, message):
    """按照统一格式打印步骤信息，使中括号后的正文起点对齐。"""
    print(f"  [{tag}]".ljust(PRINT_TAG_WIDTH + 2) + message)


# cleaned_dataset.csv 沿用原始 CSV 的前 11 列列名
FEATURE_COLUMNS = [
    "zc..",
    "alpha_c..",
    "beita_c..",
    "zb1..",
    "alpha_b1..",
    "zb2..",
    "alpha_b2..",
]
TARGET_COLUMNS = ["F1", "F2", "F3", "F4"]


class LSTMRegressor(nn.Module):
    """用于多输出时间序列回归的 LSTM 基准模型。"""

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()

        # LSTM 负责提取时间窗口内的序列特征
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 回归头将最后一个时间步的隐状态映射到 F1-F4
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.regressor(last_step)


class GRURegressor(nn.Module):
    """用于对比实验的 GRU 模型，回归头与 LSTM 保持一致。"""

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()

        # GRU 参数量少于 LSTM，可作为轻量级对比模型
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.regressor(last_step)


def make_sequences(features, targets, seq_len):
    """用过去 seq_len 个时间步的 7 维特征预测当前时刻 F1-F4。"""
    x_seq = []
    y_seq = []
    for index in range(seq_len, len(features)):
        x_seq.append(features[index - seq_len:index])
        y_seq.append(targets[index])
    return np.asarray(x_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def load_sequences_from_csv(data_path):
    """读取指定清洗数据文件，并构造时间序列样本。"""
    data = pd.read_csv(data_path)
    features = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = data[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    return make_sequences(features, targets, SEQ_LEN)


def count_parameters(model):
    """统计模型中需要训练的参数数量。"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def print_model_info(name, model):
    """打印模型结构名和可训练参数数量。"""
    print_tag("Model", name)
    print(" " * (PRINT_TAG_WIDTH + 2) + f"结构类名: {model.__class__.__name__}")
    print(" " * (PRINT_TAG_WIDTH + 2) + f"可训练参数: {count_parameters(model):,}")


def main():
    print("=" * 60)
    print("  2. 模型选择与结构设计")
    print("=" * 60)
    print("  [Step] 读取清洗后数据并构造时间序列样本")
    print("  [File] 清洗数据 : data/processed/cleaned_dataset.csv")
    print("-" * 60)

    # 读取清洗后的 7 个输入特征和 4 个输出标签
    x_seq, y_seq = load_sequences_from_csv(DATA_PATH)
    x_seq_no_smoothing, y_seq_no_smoothing = load_sequences_from_csv(NO_SMOOTHING_DATA_PATH)

    print_tag("Input", f"平滑+归一化 X: {x_seq.shape}")
    print_tag("Label", f"平滑+归一化 y: {y_seq.shape}")
    print_tag("Input", f"仅归一化 X: {x_seq_no_smoothing.shape}")
    print_tag("Label", f"仅归一化 y: {y_seq_no_smoothing.shape}")
    print_tag("Param", f"时间窗口长度: {SEQ_LEN}")
    print("-" * 60)
    print("  [Step] 构建 LSTM 基准模型与 GRU 对比模型")
    print("-" * 60)

    lstm_model = LSTMRegressor()
    gru_model = GRURegressor()
    print_model_info("LSTM 基准模型", lstm_model)
    print_model_info("GRU 对比模型", gru_model)
    print("-" * 60)

    # 用一个小批量样本检查模型输入输出维度是否符合预期
    sample_x = torch.from_numpy(x_seq[:BATCH_SIZE])
    with torch.no_grad():
        lstm_y = lstm_model(sample_x)
        gru_y = gru_model(sample_x)

    print_tag("Check", f"样本输入形状: {tuple(sample_x.shape)}")
    print_tag("Check", f"LSTM 输出形状: {tuple(lstm_y.shape)}")
    print_tag("Check", f"GRU 输出形状: {tuple(gru_y.shape)}")
    print("-" * 60)
    print("  [Done] 模型结构设计与维度检查完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
