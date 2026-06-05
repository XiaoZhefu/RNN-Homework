import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import subprocess


script_dir = os.path.dirname(os.path.abspath(__file__))
matlab_script = os.path.join(script_dir, 'matlab_data_cleansing.m').replace("\\", "/")

# 1. 数据准备 - 加载与清洗数据
print("=" * 60, flush=True)
print("  1. 数据准备：加载与清洗数据", flush=True)
print("=" * 60, flush=True)
print("  [Step] 调用 MATLAB 数据清洗脚本", flush=True)
print("  [File] MATLAB脚本 : matlab_data_cleansing.m", flush=True)
print("  [File] 原始数据   : Dateset_For_homework.csv", flush=True)
print("-" * 60, flush=True)
subprocess.run(['matlab', '-batch', f"run('{matlab_script}')"], cwd=script_dir, check=True)
print("-" * 60, flush=True)
print("  [Done] 数据清洗脚本运行结束", flush=True)
print("=" * 60, flush=True)
