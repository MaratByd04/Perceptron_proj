import numpy as np
import pickle
# import pandas as pd

def sigmoid(z):
    z = np.clip(z, -500, 500) 
    return 1 / (1 + np.exp(-z))

def deriv_sigm(a):
    return a * (1 - a)

def my_softmax(z):
    z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return z_exp / np.sum(z_exp, axis=1, keepdims=True)

def calc_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def calc_acc(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

class MLP:
    def __init__(self, layers, lr=0.01):
        self.lr = lr
        self.num = len(layers)
        self.W = []
        self.B = []

        for i in range(self.num - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.W.append(w)
            self.B.append(b)

    def forward(self, x):
        self.A = [x] # тут храним активации
        
        tmp = x
        #скрытые слои
        for i in range(self.num - 2):
            z = np.dot(tmp, self.W[i]) + self.B[i]
            tmp = sigmoid(z)
            self.A.append(tmp)
            
        z_out = np.dot(tmp, self.W[-1]) + self.B[-1]
        out = my_softmax(z_out)
        self.A.append(out)
        
        return out

    def backward(self, x, y):
        m = x.shape[0]
        pred = self.A[-1]
        dz = pred - y

        for i in range(self.num - 2, -1, -1):
            a_prev = self.A[i]
            dw = np.dot(a_prev.T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                da = np.dot(dz, self.W[i].T)
                dz = da * deriv_sigm(self.A[i])
                
            # апдейт
            self.W[i] -= self.lr * dw
            self.B[i] -= self.lr * db