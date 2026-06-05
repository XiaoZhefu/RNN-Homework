import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import subprocess

# 1. 数据准备 - 加载与清洗数据
print("=" * 60)
print("  1. 数据准备：加载与清洗数据")
print("=" * 60)
print("  [Step] 调用 MATLAB 数据清洗脚本")
print("  [File] data_cleansing.m")
print("-" * 60)
subprocess.run(['matlab', '-batch', "run('data_cleansing.m')"])
print("-" * 60)
print("  [Done] 数据清洗脚本运行结束")
print("=" * 60)
