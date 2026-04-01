# %%
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

class MPNeuron:
    def __init__(self):
        self.threshold = None
    
    def model(self, x):
        return (sum(x) >= self.threshold)

    def predict(self, X):
        Y =[]
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
    
mp_neuron = MPNeuron()
mp_neuron.threshold = 3
mp_neuron.predict([[1,1,0,1],[1,1,1,1],[0,0,0,0]])  # True


# %%
