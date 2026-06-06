import os
import subprocess


script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
matlab_script = os.path.join(project_dir, "matlab", "matlab_data_cleansing.m").replace("\\", "/")

# 1. 数据准备 - 加载与清洗数据
print("=" * 60, flush=True)
print("  1. 数据准备：加载与清洗数据", flush=True)
print("=" * 60, flush=True)
print("  [Step] 调用 MATLAB 数据清洗脚本", flush=True)
print("  [File] MATLAB脚本 : matlab/matlab_data_cleansing.m", flush=True)
print("  [File] 原始数据   : data/raw/Dateset_For_homework.csv", flush=True)
print("-" * 60, flush=True)
subprocess.run(['matlab', '-batch', f"run('{matlab_script}')"], cwd=project_dir, check=True)
print("-" * 60, flush=True)
print("  [Done] 数据清洗脚本运行结束", flush=True)
print("=" * 60, flush=True)
