# %%
from sklearn.metrics import accuracy_score
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
    
    def fit(self, X, Y):
        accuracy = {}
        for th in range(X.shape[1]+1):
            self.threshold = th
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y_pred, Y)

        self.threshold = max(accuracy, key=accuracy.get)
        print("Mejor umbral:", self.threshold)

breast_cancer = load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target

df = pd.DataFrame(data=X, columns=breast_cancer.feature_names)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(df, Y, stratify=Y)

print("Tamaño del conjunto de datos de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de datos de prueba:", X_test.shape)


X_train_bin = X_train.apply(pd.cut, bins=2, labels=[1,0])
X_test_bin = X_test.apply(pd.cut, bins=2, labels=[1,0])

# %%
mp_neuron = MPNeuron()
mp_neuron.fit(X_train_bin.values, Y_train)

#realizamos predicciones
Y_pred = mp_neuron.predict(X_test_bin.to_numpy())
print("Precisión en el conjunto de prueba:", accuracy_score(Y_test, Y_pred))

#calculamos la matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Matriz de confusión:\n", cm)