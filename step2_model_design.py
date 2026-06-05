from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "cleaned_dataset.csv"
SEQ_LEN = 50
BATCH_SIZE = 32
INPUT_SIZE = 7
OUTPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
PRINT_TAG_WIDTH = 9


def print_tag(tag, message):
    print(f"  [{tag}]".ljust(PRINT_TAG_WIDTH + 2) + message)

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
    """LSTM baseline for multi-output time series regression."""

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
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
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.regressor(last_step)


class GRURegressor(nn.Module):
    """GRU comparison model with the same regression head as LSTM."""

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()
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
    x_seq = []
    y_seq = []
    for index in range(seq_len, len(features)):
        x_seq.append(features[index - seq_len:index])
        y_seq.append(targets[index])
    return np.asarray(x_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def print_model_info(name, model):
    print_tag("Model", name)
    print(" " * (PRINT_TAG_WIDTH + 2) + f"结构类名: {model.__class__.__name__}")
    print(" " * (PRINT_TAG_WIDTH + 2) + f"可训练参数: {count_parameters(model):,}")


def main():
    print("=" * 60)
    print("  2. 模型选择与结构设计")
    print("=" * 60)
    print("  [Step] 读取清洗后数据并构造时间序列样本")
    print("  [File] 清洗数据 : cleaned_dataset.csv")
    print("-" * 60)

    data = pd.read_csv(DATA_PATH)
    features = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = data[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    x_seq, y_seq = make_sequences(features, targets, SEQ_LEN)

    print_tag("Input", f"X 形状: {x_seq.shape}")
    print_tag("Label", f"y 形状: {y_seq.shape}")
    print_tag("Param", f"时间窗口长度: {SEQ_LEN}")
    print("-" * 60)
    print("  [Step] 构建 LSTM 基准模型与 GRU 对比模型")
    print("-" * 60)

    lstm_model = LSTMRegressor()
    gru_model = GRURegressor()
    print_model_info("LSTM 基准模型", lstm_model)
    print_model_info("GRU 对比模型", gru_model)
    print("-" * 60)

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
