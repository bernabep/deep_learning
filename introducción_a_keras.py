import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras import datasets
mnist = datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

import numpy as np
import matplotlib.pyplot as plt

# Ejemplo de uso de zip()
# zip permite iterar sobre dos listas/arrays al mismo tiempo (imagen y etiqueta)
print("\nEjemplo de zip:")
for imagen, etiqueta in zip(X_train[:5], y_train[:5]):
    print(f"La imagen tiene forma {imagen.shape} y su etiqueta es: {etiqueta}")

plt.figure(figsize=(20,4))
a = zip(range(1,9),X_train[:8])
for index, digit in zip(range(1,9),X_train[:8]):
    plt.subplot(1,8,index)
    plt.imshow(np.reshape(digit,(28,28)))
    plt.title('Ejemplo ' + str(index))
# plt.show()


#vamos a dividir el conjunto de datos de prueba en 2, en prueba y y validacion
from sklearn.model_selection import train_test_split
X_test,X_val,y_test,y_val = train_test_split(X_test,y_test,test_size=0.5)
print("Tamaños de los datos:")
print(X_test.shape)
print(X_val.shape)
print(y_test.shape)
print(y_val.shape)

#ahora definimos el diseño del modelo y capas
import keras
from keras import models
from keras import layers

network = models.Sequential()
capa1= layers.Dense(300,activation='relu',input_shape=(28*28,))
capa2= layers.Dense(100,activation='relu')
capa3= layers.Dense(10,activation='softmax')
network.add(capa1)
network.add(capa2)
network.add(capa3)

print(network.summary())

#como acceder a diferentes elementos definidos
print(network.layers) 
hidden1 = network.layers[1]
weights,biases = hidden1.get_weights()
print(weights)
print(biases)

#ahora configuramos la red neuronal artificial
#-funcion de error
#-funcion de optimizacion
#-Metricas para monitorizar el proceso de entrenamiento
network.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy','Precision']
)

#preparamos el conjunto de datos
X_train_prep = X_train.reshape((60000,28*28)) #modificamos la forma de nuestra matriz
X_train_prep = X_train_prep.astype('float32') / 255 #normalizamos los datos para que trabaje mejor

X_test_prep = X_test.reshape((5000,28*28)) #modificamos la forma de nuestra matriz
X_test_prep = X_test_prep.astype('float32') / 255 #normalizamos los datos para que trabaje mejor

X_val_prep = X_val.reshape((5000,28*28)) #modificamos la forma de nuestra matriz
X_val_prep = X_val_prep.astype('float32') / 255 #normalizamos los datos para que trabaje mejor

#preparamos las caracteristicas de salida, en lugar de tener por ejemplo un 5, lo convierto en una serie de 10 digitos, donde el quinto esta marcado con un 1
from keras.utils import to_categorical
y_train_prep = to_categorical(y_train)
y_test_prep = to_categorical(y_test)
y_val_prep = to_categorical(y_val)

print(y_train[0])
print(y_train_prep[0])

#entrenamos la red neuronal artificial
history = network.fit(
    x=X_train_prep,
    y=y_train_prep,
    epochs=30,
    validation_data=(X_val_prep,y_val_prep)
                      )

#podemos mostrar en un grafico el historico del entrenamiento
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(10,7))
plt.grid(True)
plt.gca().set_ylim(0, 1.2)
plt.xlabel('epochs')
plt.show()

#validamos los datos con nuestro conjunto de datos de prueba
test_loss, test_acc, test_prec = network.evaluate(x=X_test_prep,y=y_test_prep)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
print('test_prec:', test_prec)

#ahora podemos predecir nuevos datos
X_new = X_test[34] #suponemos que es un dato nuevo
plt.imshow(np.reshape(X_new,(28,28)),cmap=plt.cm.gray)
plt.show()

X_new_prep = X_new.reshape((1,28*28))
X_new_prep = X_new_prep.astype('float32') / 255

y_proba = network.predict(X_new_prep)
print(y_proba.round(2))

#realizamos la prediccion obteniendo una clase en lugar de una probabilidad
print(np.argmax(network.predict(X_new_prep),axis=-1))

#guardamos el modelo en disco
network.save("modelo_mnist.keras")

#para poder utilizar modelos guardados hacemos lo siguiente
from keras.models import load_model
mnist_model = load_model("modelo_mnist.keras")


