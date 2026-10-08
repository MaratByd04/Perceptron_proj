import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='data.csv')
args = parser.parse_args()

print("Загружаем данные...")

df = pd.read_csv(args.dataset, header=None)

y_raw = np.where(df[1] == 'M', 1, 0)
x_raw = df.drop([0, 1], axis=1).values

y = np.zeros((len(y_raw), 2))
for i in range(len(y_raw)):
    y[i][y_raw[i]] = 1

# стандартизация руками
mean_val = np.mean(x_raw, axis=0)
std_val = np.std(x_raw, axis=0)
X = (x_raw - mean_val) / std_val

# разделение 80 на 20
np.random.seed(42)
idxs = np.random.permutation(len(X))
split_idx = int(len(X) * 0.2)

val_idx = idxs[:split_idx]
train_idx = idxs[split_idx:]

x_train = X[train_idx]
y_train = y[train_idx]
x_val = X[val_idx]
y_val = y[val_idx]

np.savez('data_train.npz', x=x_train, y=y_train)
np.savez('data_val.npz', x=x_val, y=y_val)
print("Готово. Данные сохранены.")