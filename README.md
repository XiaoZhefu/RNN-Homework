# RNN-Homework

本项目用于完成车辆振动信号到四个车轮垂向力的时间序列预测实验，包含数据预处理、LSTM/GRU 模型训练、迁移学习、模型评估和 MATLAB 可视化。

运行完整流程：

```bash
python run_all.py
```

主要目录：

- `scripts/`：Python 步骤脚本。
- `matlab/`：MATLAB 数据清洗与可视化脚本。
- `data/raw/`：原始 CSV 数据。
- `data/processed/`：清洗后的 CSV 数据。
- `models/`：训练得到的模型权重。
- `outputs/`：评估数据和输出图片。
- `report/`：LaTeX 报告及报告图片。
