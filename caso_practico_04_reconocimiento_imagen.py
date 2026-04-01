from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]

#representación visual de algunas imágenes del conjunto de datos MNIST
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')

plt.show()

#convertir a pandas
import pandas as pd
df = pd.DataFrame(X)

#creamos datos para entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=40)

#entrenamos un perceptrón
from sklearn.linear_model import Perceptron
clf = Perceptron(max_iter=2000,random_state=40, n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.coef_.shape)
print(clf.intercept_.shape)

#realizamos predicciones
y_pred = clf.predict(X_test)

#evaluamos el modelo
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1:.4f}")

#mostramos algunas predicciones erroneas
misclassified_indices = np.where(y_test != y_pred)[0]
plt.figure(figsize=(20,4))
for i, index in enumerate(misclassified_indices[:10]):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(X_test.iloc[index].values.reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test.iloc[index]}\nPred: {y_pred[index]}")
    plt.axis('off')
plt.show()
pass