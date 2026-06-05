import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import subprocess

# 1. 数据准备 - 加载与清洗数据
print("1.数据准备")
print("  正在运行MATLAB数据清洗脚本……")
subprocess.run(['matlab', '-batch', "run('data_cleansing.m')"])

