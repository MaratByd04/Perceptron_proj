import numpy as np
import argparse
import pickle
from network import MLP, calc_loss, calc_acc

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data_val.npz')
parser.add_argument('--model_file', default='saved_model.npy')
args = parser.parse_args()

try:
    val_data = np.load(args.data)
    x_val = val_data['x']
    y_val = val_data['y']
except:
    print("Не смог открыть данные")
    exit()

#создаем пустую сеть
net = MLP([x_val.shape[1], 2]) 

with open(args.model_file, 'rb') as f:
    data = pickle.load(f)
net.W = data['W']
net.B = data['B']
net.num = len(net.W) + 1

res = net.forward(x_val)
l = calc_loss(y_val, res)
a = calc_acc(y_val, res)

print(f"Loss: {l:.4f}")
print(f"Accuracy: {a:.4f}")