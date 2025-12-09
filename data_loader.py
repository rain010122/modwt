import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path, target_col='OT', scale=True):
    """
    加载并预处理数据
    :param file_path: csv 文件路径
    :param target_col: 主要分析的目标列 (如 OT)，如果是 None 则分析所有列的平均值
    :return: 处理后的 numpy 数组
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # 剔除非数值列 (通常是 date)
    cols_to_drop = ['date']
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    data = df.values

    # 标准化 (Z-Score)
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    # 策略：如果是多变量，我们计算“全局主导周期”
    # 方法是：计算每一列的小波谱，然后取平均？
    # 或者简单起见，这里只取目标列 OT 做演示。
    # 如果想更严谨，可以取 data[:, -1] (OT列) 或者 np.mean(data, axis=1)
    
    if target_col and target_col in df.columns:
        target_idx = df.columns.get_loc(target_col)
        series = data[:, target_idx]
    else:
        # 如果没有指定列，或者找不到列，取最后一列作为代表
        series = data[:, -1]
        
    return series