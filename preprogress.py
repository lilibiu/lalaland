import pandas as pd
import numpy as np


def generate_data(data_path, time_step):
    data = pd.read_csv(data_path, encoding='gbk')
    data = np.array(data[['收盘价', '最高价', '最低价', '开盘价']])

    data = data[::-1]
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    norm_data = (data - data_mean) / data_std

    X = []
    y = []
    for i in range(len(norm_data)-time_step):
        X.append(norm_data[i:i+time_step])
        y.append(norm_data[i+time_step])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(data_mean), np.array(data_std)


if __name__ == "__main__":
    X, y, data_mean, data_std = generate_data("F:/AnaWork/lstm_stock/data/SH000001_2_train.csv", 30)
    print(X[:3])
    print(y[:3])
    print(np.shape(X), np.shape(y))