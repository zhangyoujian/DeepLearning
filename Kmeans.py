import numpy as np
import matplotlib.pyplot as plt

class Kmeans():
    def __init__(self):
        self.centralPoint = None

    def initialCentral(self, X, K):

        np.random.seed()
        M = list(range(X.shape[0]))
        np.random.shuffle(M)

        selectK = M[0:K]
        centralPoint = X[selectK, :]

        self.centralPoint = centralPoint
        return centralPoint

    def computeDistance(self, centralPoint, X):

        K = centralPoint.shape[0]
        N = X.shape[0]
        Dist = np.zeros((N,K),dtype=np.float)
        for i in range(K):
            d = np.sqrt(np.sum(np.power(centralPoint[i,:] - X, 2), axis = 1))
            Dist[:, i] = d

        return Dist

    def updateCentral(self,Dist, X):

        N = X.shape[0]
        K = Dist.shape[1]
        index = np.argmin(Dist, axis=1)
        centralPoint = np.zeros((K,X.shape[1]), dtype=np.float)
        Loss = 0
        for i in range(K):
            P = np.mean(X[index == i, :], axis=0, keepdims=True)
            centralPoint[i, :] = P
            Loss += np.sum(np.power(X[index == i, :] - P, 2))

        self.centralPoint = centralPoint
        Loss = Loss / N
        return Loss


    def fit(self, X, K, epoch = 10):

        self.initialCentral(X, K)
        Loss = []
        for i in range(epoch):
            Dist = self.computeDistance(self.centralPoint, X)
            min_loss = self.updateCentral(Dist, X)
            Loss.append(min_loss)

            print("iterative[%d] Loss is %2.3f"%(i+1, min_loss))


        plt.figure(0)
        plt.plot(range(1,epoch+1),Loss,lw=2)
        plt.xlabel('iterative')
        plt.ylabel('Loss')

        if K==3 and X.shape[1]>=2:
            Dist = self.computeDistance(self.centralPoint,X)
            index = np.argmin(Dist,axis=1)
            Cluster1 = X[index==0,:]
            Cluster2 = X[index==1,:]
            Cluster3 = X[index==2,:]

            plt.figure(1)
            plt.scatter(Cluster1[:, 0], Cluster1[:, 1],10, c='r', marker='o',label='Cluster 1')
            plt.scatter(Cluster2[:, 0], Cluster2[:, 1], 10,c='b', marker='o', label='Cluster 2')
            plt.scatter(Cluster3[:, 0], Cluster3[:, 1], 10, c='k', marker='o', label='Cluster 3')
            plt.xlabel('feature 1', fontsize=18)
            plt.ylabel('feature 2', fontsize=18)

        plt.show()

    def predict(self, X):

        Dist = self.computeDistance(self.centralPoint, X)
        pred = np.argmin(Dist, axis=1)
        return pred

