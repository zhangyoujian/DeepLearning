import numpy as np
import copy

def lineaActive(z):
    return z

def lineaActive_derive(z):
    ret = np.ones(z.shape)
    return ret

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derive(z):
    return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
    return np.tanh(z)

def tanh_derive(z):
    return 1.0-np.power(tanh(z),2)

def ReLu(z):
    return np.maximum(z,0.0)

def ReLu_derive(z):
    z[z<=0] =0
    z[z>0] = 1
    return z

class FullLayer(object):
    def __init__(self,inputShape,outputshape,active_fun,eta):
        self.inputShape = inputShape
        self.outputshape = outputshape
        self.eta = eta
        if active_fun=='sigmoid':
            self.active_fun = sigmoid
            self.active_derive_fun = sigmoid_derive
        elif active_fun=='tanh':
            self.active_fun = tanh
            self.active_derive_fun = tanh_derive
        elif active_fun=='ReLu':
            self.active_fun = ReLu
            self.active_derive_fun = ReLu_derive
        elif active_fun=='lineaActive':
            self.active_fun = lineaActive
            self.active_derive_fun = lineaActive_derive
        else:
            self.active_fun = lineaActive
            self.active_derive_fun = lineaActive_derive


        self.weight = np.array(
            np.random.normal(loc=0, scale=np.sqrt(1 / inputShape), size=(outputshape,inputShape))
        )
        self.bias = np.array(
            np.random.normal(loc=0, scale=1, size=(outputshape, 1))
        )

        self.delta_weight = 0
        self.delta_bias = 0

        self.active = 0
        self.z = 0
        self.error = 0


class softmax(object):
    def __init__(self,inputShape,outputshape,eta):
        self.inputShape = inputShape
        self.outputshape = outputshape
        self.eta = eta

        self.weight = np.array(
            np.random.normal(loc=0, scale=np.sqrt(1 / inputShape), size=(outputshape, inputShape))
        )
        self.bias = np.array(
            np.random.normal(loc=0, scale=1, size=(outputshape, 1))
        )

        self.delta_weight = 0
        self.delta_bias = 0
        self.active = 0
        self.z = 0
        self.error = 0



class network(object):
    def __init__(self,layers,trainningdata,ValidationData,TestingData):
        self.layer = layers
        self.trainningdata = trainningdata
        self.ValidationData = ValidationData
        self.TestingData = TestingData

        X = []
        Y = []
        for x, y in self.trainningdata:
            X.append(x)
            Y.append(y)
        self.traingX = np.array(X)
        self.traingY = np.array(Y)

        self.active0 = 0

    def forward(self,item):
        self.active0 = self.traingX[item]  # [1 28 28]


        #=========================通过隐藏层========================
        self.layer[0].z = np.dot(self.layer[0].weight,self.active0) + self.layer[0].bias
        self.layer[0].active = self.layer[0].active_fun(self.layer[0].z)
        #===========================通过softmax输出层===============================
        self.layer[1].z = np.dot(self.layer[1].weight,  self.layer[0].active) + self.layer[1].bias
        self.layer[1].active = np.exp(self.layer[1].z)/np.sum(np.exp(self.layer[1].z),axis=None)

        yhat = self.layer[1].active
        y = self.traingY[item]

        cost = -y*np.log(yhat)
        return np.sum(cost)

    def backprob(self,item):

        error =  self.layer[1].active - self.traingY[item]

        e = error
        z = self.layer[0].active

        self.layer[1].delta_weight = self.layer[1].delta_weight + np.dot(e,z.transpose())
        self.layer[1].delta_bias = self.layer[1].delta_bias  + e
        self.layer[1].error = np.dot(self.layer[1].weight.transpose(),e)


        self.layer[0].error = np.multiply(self.layer[1].error, self.layer[0].active_derive_fun(self.layer[0].z))

        self.layer[0].delta_weight = self.layer[0].delta_weight + np.dot(self.layer[0].error,self.active0.transpose())
        self.layer[0].delta_bias = self.layer[0].delta_bias + self.layer[0].error


    def updateWeight(self,m):

        self.layer[0].weight =  self.layer[0].weight - self.layer[0].eta/m*self.layer[0].delta_weight
        self.layer[0].bias = self.layer[0].bias - self.layer[0].eta/m*self.layer[0].delta_bias

        self.layer[1].weight = self.layer[1].weight - self.layer[1].eta / m * self.layer[1].delta_weight
        self.layer[1].bias = self.layer[1].bias - self.layer[1].eta / m * self.layer[1].delta_bias

        self.layer[0].delta_weight = 0
        self.layer[0].delta_bias = 0
        self.layer[1].delta_weight = 0
        self.layer[1].delta_bias = 0

    def SGD(self):
        m,n,c = self.traingX.shape
        cost = 0.0
        for item in range(m):
            cost = cost + self.forward(item)
            self.backprob(item)
        cost = cost / m
        print('cost is :{0}'.format(cost))
        self.updateWeight(m)

    def predicted(self):
        m, nh,c = self.traingY.shape
        count = 0
        for i in range(m):
            self.forward(i)
            y = self.layer[1].active
            yhat = self.traingY[i]
            if np.argmax(y) == np.argmax(yhat):
                count = count + 1
        return count, m









