import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class NeuralNetwork:
    def __init__(self, X:np.array, Y:np.array, lr=0.0001):
        """
        初始化neuralnetwork对象
        :param X: 矩阵维度为(n, m), 其中n为数据维度，m为样本数量
        :param Y: 矩阵维度为(p, m), 其中m为样本数量
        :return:
        """
        self.X = X
        self.Y = Y

        self.lr = lr

        ni = X.shape[0]
        no = Y.shape[0]
        nl = math.floor(math.sqrt(ni + no)) + 3

        hide_layer_weight = np.random.rand(nl, ni) * 0.01
        out_layer_weight = np.random.rand(no, nl) * 0.01

        hide_layer_bias = np.random.rand(nl, 1) * 0.01
        out_layer_bias = np.random.random() * 0.01
        self.w = [hide_layer_weight, out_layer_weight]
        self.b = [hide_layer_bias, out_layer_bias]

    def __call__(self, x:np.array)->np.array:
        predict_tuple = self.forward(x)
        return predict_tuple

    @staticmethod
    def sigmod(z: np.array) -> np.array:
        yhat = 1.0 / (1.0 + np.exp(-z))
        return np.clip(yhat, 1e-8, 1 - 1e-8)  # 限制在[1e-8, 1-1e-8]

    @staticmethod
    def tanh(z: np.array)-> np.array:
        yhat = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return yhat

    @staticmethod
    def tanh_derive(z):
        out = 1.0 - np.power(NeuralNetwork.tanh(z), 2)
        return out

    def forward(self, x:np.array)-> np.array:
        z_1 = np.dot(self.w[0], x) + self.b[0]
        a_1 = self.tanh(z_1)

        z_2 = np.dot(self.w[1], a_1) + self.b[1]
        a_2 = self.sigmod(z_2)

        return z_1, a_1, z_2, a_2

    def backpropagation(self, predict_list:tuple):
        z_1 = predict_list[0]                   # 维度:  (l, m）
        a_1 = predict_list[1]                   # 维度:  (l, m)
        z_2 = predict_list[2]                   # 维度:  (1, m)
        yhat = predict_list[3]                  # 维度:  (1, m)

        m = self.Y.shape[1]
        error_o = yhat - self.Y                     # 输出层误差 (1, m)
        #                             (l, 1)  *  (1, m)       X     (l, m)
        error_l = np.multiply(np.dot(self.w[1].T, error_o), self.tanh_derive(z_1))  # 隐藏层误差 (l, m)

        dw = [np.dot(error_l, self.X.T),
              np.dot(error_o, a_1.T)]

        db = [np.sum(error_l, axis=1, keepdims=True),
              np.sum(error_o)]

        # 更新权重
        self.w[0] -= self.lr * 1.0 / m * dw[0]
        self.w[1] -= self.lr * 1.0 / m * dw[1]

        # 更新偏置
        self.b[0] -= self.lr * 1.0 / m * db[0]
        self.b[1] -= self.lr * 1.0 / m * db[1]

    def train(self, epochs):
        for _ in range(epochs):
            predict_list = self.forward(self.X)
            self.backpropagation(predict_list)
            loss = self.LossFunc(predict_list[-1])
            print(f"Epoch {_ + 1}, Loss: {loss:.4f}")

    def LossFunc(self, yhat)->float:
        m = self.Y.shape[1]
        # 内部二次裁剪确保稳定
        yhat = np.clip(yhat, 1e-8, 1 - 1e-8)
        cost = -1.0 / m * (np.sum(self.Y * np.log(yhat) + (1 - self.Y) * np.log(1 - yhat)))  # 返回标量
        return cost

def main():
    iris = load_iris()
    # 将三种类型的花转为两种类别
    mask = (iris.target == 0) | (iris.target == 1)

    X, Y = iris.data[mask], iris.target[mask].reshape(1, -1)
    X = X.T
    model = NeuralNetwork(X, Y, 0.001)
    total_epoch = 100000
    loss_curve = []
    for epoch in range(total_epoch):
        predict_tuple = model(X)
        model.backpropagation(predict_tuple)
        loss = model.LossFunc(predict_tuple[-1])
        loss_curve.append(loss)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    plt.figure()
    plt.plot(range(total_epoch), loss_curve, 'b--', linewidth=2, label='Train Loss')
    plt.title('Binary nerualnetwork Training Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__=='__main__':
    main()