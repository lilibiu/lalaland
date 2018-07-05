import pandas as pd
import numpy as np


def generate_data(data_path, time_step):
    data = pd.read_csv(data_path, encoding='gbk')
    data = np.array(data['收盘价'])
    data = data[::-1]

    data_mean = np.mean(data)
    data_std = np.std(data)
    normalize_data = (data - data_mean) / data_std

    X = []
    y = []
    for i in range(len(normalize_data)-time_step):
        X.append([normalize_data[i:i+time_step]])
        y.append([normalize_data[i+time_step]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), data_mean, data_std


if __name__ == "__main__":
    X, y = generate_data("F:\AnaWork\SH000001.csv", 30)
    print(X[:3])
    print(y[:3])
    print(X.dtype,y.dtype)