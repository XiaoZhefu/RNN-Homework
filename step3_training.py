from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from step2_model_design import (
    BATCH_SIZE,
    DATA_PATH,
    FEATURE_COLUMNS,
    GRURegressor,
    LSTMRegressor,
    PRINT_TAG_WIDTH,
    SEQ_LEN,
    TARGET_COLUMNS,
    make_sequences,
    print_tag,
)


# 输出目录：模型权重与训练损失会分别保存到这里
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "outputdata"

# 训练超参数
TRAIN_RATIO = 0.8
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42


def set_random_seed(seed):
    """固定随机种子，使训练过程尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sequence_data():
    """读取清洗后的数据，并转换成 RNN 需要的三维时间序列样本。"""
    data = pd.read_csv(DATA_PATH)
    features = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = data[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    return make_sequences(features, targets, SEQ_LEN)


def split_train_val(x_seq, y_seq, train_ratio):
    """按时间顺序划分训练集和验证集，避免时间序列信息泄漏。"""
    train_size = int(len(x_seq) * train_ratio)
    return (
        x_seq[:train_size],
        y_seq[:train_size],
        x_seq[train_size:],
        y_seq[train_size:],
    )


def build_dataloader(x_data, y_data, batch_size, shuffle):
    """将 NumPy 数组封装成 PyTorch DataLoader。"""
    dataset = TensorDataset(torch.from_numpy(x_data), torch.from_numpy(y_data))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch，并返回该 epoch 的平均训练损失。"""
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


def validate_one_epoch(model, dataloader, criterion, device):
    """在验证集上计算平均损失，不进行梯度更新。"""
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y)

            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / total_count


def train_model(model_name, model, train_loader, val_loader, device):
    """训练指定模型，使用验证损失保存最优权重，并通过早停防止过拟合。"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model = model.to(device)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_count = 0
    history = []
    model_path = MODEL_DIR / f"{model_name.lower()}_best.pt"

    print_tag("Step", f"训练 {model_name} 模型")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            " " * (PRINT_TAG_WIDTH + 2)
            + f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            + f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        # 验证损失下降则保存当前最优模型，否则累积早停计数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print_tag("Early", f"{model_name} 在第 {epoch} 轮触发早停")
                break

    print_tag("Best", f"{model_name} 最优轮次: {best_epoch}, 验证损失: {best_val_loss:.6f}")
    return history


def main():
    print("=" * 60)
    print("  3. 模型训练")
    print("=" * 60)
    print("  [Step] 划分训练集与验证集")
    print("  [File] 清洗数据 : cleaned_dataset.csv")
    print("-" * 60)

    set_random_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    x_seq, y_seq = load_sequence_data()
    x_train, y_train, x_val, y_val = split_train_val(x_seq, y_seq, TRAIN_RATIO)
    train_loader = build_dataloader(x_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(x_val, y_val, BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_tag("Input", f"训练集 X: {x_train.shape}, y: {y_train.shape}")
    print_tag("Label", f"验证集 X: {x_val.shape}, y: {y_val.shape}")
    print_tag("Param", f"批量大小: {BATCH_SIZE}, 学习率: {LEARNING_RATE}")
    print_tag("Param", f"最大轮数: {MAX_EPOCHS}, 早停耐心值: {PATIENCE}")
    print_tag("Param", f"优化器: Adam, 权重衰减: {WEIGHT_DECAY}")
    print_tag("Device", str(device))
    print("-" * 60)

    all_history = []
    all_history.extend(train_model("LSTM", LSTMRegressor(), train_loader, val_loader, device))
    print("-" * 60)
    all_history.extend(train_model("GRU", GRURegressor(), train_loader, val_loader, device))

    loss_path = OUTPUT_DIR / "loss_history.csv"
    pd.DataFrame(all_history).to_csv(loss_path, index=False, encoding="utf-8-sig")

    print("-" * 60)
    print_tag("File", f"训练损失: {loss_path}")
    print_tag("File", f"模型权重: {MODEL_DIR}")
    print("-" * 60)
    print("  [Done] 模型训练完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
