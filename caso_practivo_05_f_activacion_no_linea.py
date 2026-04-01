#Detección de transacciones bancarias fraudulentas
#realizamos importaciones de librerías necesarias para el desarrollo del proyecto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#creo funciones auxiliares para el desarrollo del proyecto
# Construcción de una función que realice el particionado completo
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train, test = train_test_split(df, test_size=0.2, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test[stratify] if stratify else None
    val_set, test_set = train_test_split(test, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return X, y

#representación del límite de decisión de una red neuronal
def plot_ann_decision_boundary(X, y, model, steps=1000):
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1 #agregamos un margen de 1 a los límites para mejorar la visualización
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1 #agregamos un margen de 1 a los límites para mejorar la visualización
    x_span = np.linspace(x_min, x_max, steps) #creamos un espacio de puntos entre los límites mínimos y máximos de la característica x
    y_span = np.linspace(y_min, y_max, steps) #creamos un espacio de puntos entre los límites mínimos y máximos de la característica y
    xx, yy = np.meshgrid(x_span, y_span) #creamos una malla de puntos a partir de los espacios creados para las características x e y
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()]) #obtenemos las predicciones del modelo para cada punto de la malla
    Z = labels.reshape(xx.shape) #reformamos las predicciones para que tengan la misma forma que la malla

    plt.contourf(xx, yy, Z, cmap="RdBu", alpha=0.5) #dibujamos el límite de decisión utilizando un mapa de colores
    plt.plot(X[:,0][y==0], X[:,1][y==0], 'k.', markersize=2) #dibujamos los puntos de la clase 0 en verde
    plt.plot(X[:,0][y==1], X[:,1][y==1], 'r.', markersize=2) #dibujamos los puntos de la clase 1 en rojo
    plt.xlabel('V10',fontsize=12) #etiqueta del eje x
    plt.ylabel('V14',fontsize=12) #etiqueta del eje y
    


#cargamos el conjunto de datos
df = pd.read_csv("src/creditcard.csv")
print(df.head())

#representación grafica de dos características del conjunto de datos
plt.figure(figsize=(14, 6))
plt.scatter(df['V10'][df['Class'] == 0], df['V14'][df['Class'] == 0], c='g', marker='.')
plt.scatter(df['V10'][df['Class'] == 1], df['V14'][df['Class'] == 1], c='r', marker='.')

plt.xlabel('V10')
plt.ylabel('V14')
plt.legend()
plt.show()

#dividimos el conjunto de datos en entrenamiento, validación y prueba
train, val, test = train_val_test_split(df)

X_train, y_train = remove_labels(train, 'Class')
X_val, y_val = remove_labels(val, 'Class')
X_test, y_test = remove_labels(test, 'Class')

# reducimos el conjunto de datos a solo dos características para facilitar la visualización
X_train_reduced = X_train[['V10', 'V14']].copy()
X_val_reduced = X_val[['V10', 'V14']].copy()
X_test_reduced = X_test[['V10', 'V14']].copy()

#definimos arquitectura de la red neuronal
from keras import models, layers
activation = None # si dejamos activado el valor None, la función de activación será lineal, lo que significa que la salida de cada neurona será una combinación lineal de sus entradas. Esto puede limitar la capacidad del modelo para aprender patrones complejos en los datos. Al establecer activation a 'relu', estamos utilizando la función de activación ReLU (Rectified Linear Unit), que introduce no linealidad en el modelo y permite que aprenda patrones más complejos. La función ReLU devuelve 0 para valores negativos y el valor de entrada para valores positivos, lo que ayuda a evitar el problema del gradiente desaparecido y mejora el rendimiento del modelo.
activation = 'relu' # La función de activación ReLU (Rectified Linear Unit) es una función no lineal que se utiliza comúnmente en redes neuronales. Devuelve 0 para valores negativos y el valor de entrada para valores positivos. Esto introduce no linealidad en el modelo, lo que permite que aprenda patrones más complejos en los datos. Además, la función ReLU ayuda a evitar el problema del gradiente desaparecido, lo que mejora el rendimiento del modelo durante el entrenamiento.

model = models.Sequential()
model.add(layers.Input(shape=(X_train_reduced.shape[1],)))
model.add(layers.Dense(128, activation=activation, input_shape=(X_train_reduced.shape[1],)))
model.add(layers.Dense(64, activation=activation))
model.add(layers.Dense(32, activation=activation))
model.add(layers.Dense(16, activation=activation))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy','Precision'])

#resumen de la arquitectura de la red neuronal
print(model.summary())

history = model.fit(X_train_reduced, 
                    y_train, 
                    epochs=30, 
                    validation_data=(X_val_reduced, y_val))

plt.figure(figsize=(14, 6))
plot_ann_decision_boundary(X_train_reduced.values, y_train.values, model)
plt.title('Límite de decisión de la red neuronal', fontsize=16)
plt.show()
