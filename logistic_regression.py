import numpy as np
import matplotlib.pyplot as plt

# 二元logistic回归实现

class Logistic:
    def __init__(self, X:np.array, Y:np.array, lr=0.01):
        """
        初始化Logistic对象
        :param X: 矩阵维度为(n, m), 其中n为数据维度，m为样本数量
        :param Y: 矩阵维度为(1, m), 其中m为样本数量
        :return:
        """
        self.X = X
        self.Y = Y.reshape(1, -1)  # 强制转为(1, m)

        # w为权重
        self.w = np.random.rand(X.shape[0], 1) * 0.01
        # b为偏置
        self.b = np.zeros((0, 0))
        # lr为学习速率
        self.lr = lr

    def __call__(self, x:np.array)->np.array:
        output = self.forward(x)
        return output

    @staticmethod
    def sigmod(z:np.array)->np.array:
        yhat = 1.0 / (1.0 + np.exp(-z))
        return np.clip(yhat, 1e-8, 1-1e-8)  # 限制在[1e-8, 1-1e-8]

    def forward(self, x:np.array)->np.array:
        z = np.dot(self.w.transpose(), x) + self.b
        yhat = self.sigmod(z)
        return yhat

    def backpropagation(self, yhat:np.array):
        m = self.Y.shape[1]
        error = yhat - self.Y         # 维度(1, m)

        dw = np.dot(self.X, error.T)  # 维度(n, 1)
        db = np.sum(error)            # 标量

        self.w -= self.lr * 1.0 / m * dw
        self.b -= self.lr * 1.0 / m * db

    def LossFunc(self, yhat)->float:
        m = self.Y.shape[1]
        # 内部二次裁剪确保稳定
        yhat = np.clip(yhat, 1e-8, 1 - 1e-8)
        cost = -1.0 / m * (np.sum(self.Y * np.log(yhat) + (1 - self.Y) * np.log(1 - yhat)))  # 返回标量
        return cost

def main():
    epoch = 100000
    model = Logistic(None, None)
    for i in range(epoch):



if __name__=='__main__':
    main()
