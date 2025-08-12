import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, X:np.array, Y:np.array, hideLayerSize=7, lr=0.01):
        self.X = X
        self.Y = Y

        self.hideLayerSize = hideLayerSize
        self.lr = lr

        