import pandas as pd
import torch
from step2_model_design import (
    BATCH_SIZE,
    GRURegressor,
    LSTMRegressor,
    print_tag,
)
from step3_training import (
    DATASET_CONFIGS,
    MODEL_DIR,
    OUTPUT_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    build_dataloader,
    load_sequence_data,
    set_random_seed,
    split_train_val_test,
    train_model,
    RANDOM_SEED,
)


# 迁移学习设置：源域为平滑+归一化，目标域为仅归一化
SOURCE_MODEL_PREFIX = DATASET_CONFIGS[0]["model_prefix"]
TARGET_DATA_CONFIG = DATASET_CONFIGS[1]
TRANSFER_MODEL_PREFIX = "transfer_"
TRANSFER_LOSS_FILE = "loss_history_transfer.csv"


def load_pretrained_model(model_name, device):
    """加载源域训练得到的模型权重，作为迁移学习的初始化参数。"""
    if model_name == "LSTM":
        model = LSTMRegressor()
    elif model_name == "GRU":
        model = GRURegressor()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pretrained_path = MODEL_DIR / f"{SOURCE_MODEL_PREFIX}{model_name.lower()}_best.pt"
    if not pretrained_path.exists():
        raise FileNotFoundError(f"缺少源域预训练权重: {pretrained_path}")

    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    print_tag("File", f"{model_name} 预训练权重: {pretrained_path.name}")
    return model


def main():
    print("=" * 60)
    print("  4. 迁移学习")
    print("=" * 60)
    print("  [Step] 加载平滑+归一化模型权重，并迁移到仅归一化数据")
    print("  [File] 源域权重 : models/lstm_best.pt, models/gru_best.pt")
    print("  [File] 目标数据 : data/processed/cleaned_dataset_no_smoothing.csv")
    print("-" * 60)

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_seq, y_seq = load_sequence_data(TARGET_DATA_CONFIG["data_path"])
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(
        x_seq,
        y_seq,
        TRAIN_RATIO,
        VAL_RATIO,
    )
    train_loader = build_dataloader(x_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(x_val, y_val, BATCH_SIZE, shuffle=False)

    print_tag("Input", f"目标训练集 X: {x_train.shape}, y: {y_train.shape}")
    print_tag("Label", f"目标验证集 X: {x_val.shape}, y: {y_val.shape}")
    print_tag("Test", f"目标测试集 X: {x_test.shape}, y: {y_test.shape}")
    print_tag("Device", str(device))
    print("-" * 60)

    all_history = []
    for model_name in ["LSTM", "GRU"]:
        set_random_seed(RANDOM_SEED)
        model = load_pretrained_model(model_name, device)
        all_history.extend(
            train_model(
                model_name,
                model,
                TRANSFER_MODEL_PREFIX,
                train_loader,
                val_loader,
                device,
            )
        )
        print("-" * 60)

    loss_path = OUTPUT_DIR / TRANSFER_LOSS_FILE
    pd.DataFrame(all_history).to_csv(loss_path, index=False, encoding="utf-8-sig")
    print_tag("File", f"迁移训练损失: {loss_path}")
    print_tag("File", f"迁移模型权重: {MODEL_DIR}")
    print("-" * 60)
    print("  [Done] 迁移学习完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
