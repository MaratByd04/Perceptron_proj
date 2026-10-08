import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from network import MLP, calc_loss, calc_acc

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', default='data_train.npz')
parser.add_argument('--val_data', default='data_val.npz')
parser.add_argument('--layer', nargs='+', type=int, default=[24, 24])
parser.add_argument('--epochs', type=int, default=84)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=0.0314)
args = parser.parse_args()

train_file = np.load(args.train_data)
val_file = np.load(args.val_data)

x_train = train_file['x']
y_train = train_file['y']
x_val = val_file['x']
y_val = val_file['y']

in_size = x_train.shape[1]
out_size = 2
my_layers = [in_size] + args.layer + [out_size]

print(f"x_train shape: {x_train.shape}")
print(f"x_valid shape: {x_val.shape}")

net = MLP(my_layers, lr=args.learning_rate)

loss_hist = []
val_loss_hist = []
acc_hist = []
val_acc_hist = []

m = len(x_train)

for ep in range(args.epochs):              # цикл обучения
    perm = np.random.permutation(m)
    x_tr = x_train[perm]
    y_tr = y_train[perm]
    
    for i in range(0, m, args.batch_size):
        xb = x_tr[i:i+args.batch_size]
        yb = y_tr[i:i+args.batch_size]
        net.forward(xb)
        net.backward(xb, yb)
        
    p_train = net.forward(x_train)
    tr_l = calc_loss(y_train, p_train)
    tr_a = calc_acc(y_train, p_train)
    
    p_val = net.forward(x_val)
    v_l = calc_loss(y_val, p_val)
    v_a = calc_acc(y_val, p_val)
    
    loss_hist.append(tr_l)
    val_loss_hist.append(v_l)
    acc_hist.append(tr_a)
    val_acc_hist.append(v_a)
    
    print(f"epoch {ep+1}/{args.epochs} loss: {tr_l:.4f} val_loss: {v_l:.4f}")

# сохраняем
with open('saved_model.npy', 'wb') as f:
    pickle.dump({'W': net.W, 'B': net.B}, f)
print("> saving model './saved_model.npy' to disk...")

# рисуем графики
epochs_range = range(1, args.epochs + 1)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss_hist, label='training loss')
plt.plot(epochs_range, val_loss_hist, label='validation loss', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc_hist, label='training acc')
plt.plot(epochs_range, val_acc_hist, label='validation acc')
plt.legend()
plt.grid(True)

plt.show()