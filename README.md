# RNN-Homework

本项目用于完成车辆振动信号到四个车轮垂向力的时间序列预测实验，包含数据清洗、LSTM/GRU 模型设计、模型训练、迁移学习、模型评估和 MATLAB 可视化。

## 项目结构

```text
RNN-Homework/
├─ .gitignore
├─ LICENSE
├─ README.md
├─ RNN-Homework.code-workspace
├─ run_all.py
│
├─ data/
│  ├─ processed/                         # 运行后生成/更新：清洗后的数据
│  │  ├─ cleaned_dataset.csv
│  │  └─ cleaned_dataset_no_smoothing.csv
│  └─ raw/
│     └─ Dateset_For_homework.csv
│
├─ matlab/
│  ├─ matlab_data_cleansing.m
│  └─ matlab_visualization_evaluation.m
│
├─ models/                               # 运行后生成/更新：模型权重
│  ├─ gru_best.pt
│  ├─ lstm_best.pt
│  ├─ no_smoothing_gru_best.pt
│  ├─ no_smoothing_lstm_best.pt
│  ├─ transfer_gru_best.pt
│  └─ transfer_lstm_best.pt
│
├─ outputs/                              # 运行后生成/更新：实验输出
│  ├─ data/
│  │  ├─ evaluation_metrics.csv
│  │  ├─ evaluation_metrics_no_smoothing.csv
│  │  ├─ evaluation_metrics_transfer.csv
│  │  ├─ loss_history.csv
│  │  ├─ loss_history_no_smoothing.csv
│  │  ├─ loss_history_transfer.csv
│  │  ├─ prediction_results.csv
│  │  ├─ prediction_results_no_smoothing.csv
│  │  └─ prediction_results_transfer.csv
│  └─ pics/
│     └─ ...（略）
│
├─ report/
│  ├─ RNN_LaTeX_Report.tex
│  ├─ out/                               # 运行后生成/更新：LaTeX 编译产物
│  │  ├─ RNN_LaTeX_Report.pdf
│  │  └─ ...（略）
│  └─ pics/                              # 运行后生成/更新：报告引用图片
│     └─ ...（略）
│
└─ scripts/
   ├─ step1_preparing.py
   ├─ step2_model_design.py
   ├─ step3_training.py
   ├─ step4_transfer_learning.py
   └─ step5_evaluation.py
```

## 脚本用法

在项目根目录运行总控脚本：

```bash
python run_all.py
```

总控脚本会依次调用 `scripts/` 下的 5 个步骤，并根据已有文件自动跳过已经完成的阶段：

- `step1_preparing.py`：调用 MATLAB 清洗原始数据，生成 `data/processed/` 下的 CSV，并输出预处理对比图。
- `step2_model_design.py`：定义输入输出变量、时间窗口、LSTM/GRU 网络结构和模型参数。
- `step3_training.py`：训练 LSTM/GRU，并保存普通训练和仅归一化对比实验的权重与损失记录。
- `step4_transfer_learning.py`：加载平滑+归一化数据上训练得到的权重，在仅归一化数据上继续训练。
- `step5_evaluation.py`：在测试集上计算 MSE、RMSE、MAE，保存预测结果，并调用 MATLAB 绘制结果图。

如果只想调试某一步，也可以在项目根目录下单独运行：

```bash
python scripts/step3_training.py
```

报告位于 `report/RNN_LaTeX_Report.tex`，编译产物建议输出到 `report/out/`。
