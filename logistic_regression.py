import numpy as np
import matplotlib.pyplot as plt




class logistic():
    def __init__(self):
        self.L2 = 0
        self.Z = 0
        self.A = 0

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    def intialParameter(self, hidelayer):
        self.W = np.random.rand(hidelayer, 1)
        self.b = np.random.rand()


    def feedforward(self, X):

        self.X = X
        self.Z = np.dot(self.X, self.W) + self.b
        self.A = self.sigmoid(self.Z)
        self.A = self.A[:,0]
        return self.A

    def costFun(self,ylabel):

        yhat = self.A
        M = ylabel.shape[0]
        cost = ylabel * np.log(yhat) + (1 - ylabel) * np.log(1-yhat)
        cost = -1 / M * np.sum(cost) + self.L2 / (2 * M) * np.sum(np.power(self.W, 2))
        return cost

    def backforward(self, ylabel):

        M = ylabel.shape[0]

        delta = (self.A - ylabel)/M
        delta = delta.reshape((-1,1))
        delta_w = np.dot(self.X.transpose(), delta) + self.L2 / M * self.W
        delta_b = np.sum(delta)

        self.delta_w = delta_w
        self.delta_b = delta_b


    def updateWeights(self, learning_rate):
        self.W = self.W - learning_rate * self.delta_w
        self.b = self.b - learning_rate * self.delta_b

    def predict(self, X):

        pred = self.feedforward(X)
        P = np.zeros(pred.shape[0], dtype=np.int)
        P[pred>=self.threshhold] = 1
        return P


    def fit(self, X, Y, learning_rate = 0.1, L2 = 0.001, epoch = 5000, threshhold = 0.5):

        self.threshhold = threshhold
        self.L2 = L2
        M = X.shape[0]
        N = X.shape[1]

        self.intialParameter(N)
        Loss = []
        for i in range(epoch):
            self.feedforward(X)
            loss = self.costFun(Y)
            Loss.append(loss)
            self.backforward(Y)
            self.updateWeights(learning_rate)

            print('iterative[%d] loss is %3.4f'%(i+1, loss))


        plt.figure(0)
        plt.plot(range(1,epoch+1), Loss,c='r', lw=2)
        plt.xlabel('iterative')
        plt.ylabel('Loss')
        plt.show()

        Pred = self.predict(X)
        ylabel = np.round(Y)

        Accuracy = np.mean(Pred==ylabel)*100
        print("Accuracy is %3.2f"%(Accuracy))
        print(self.b)
        print(self.W)











