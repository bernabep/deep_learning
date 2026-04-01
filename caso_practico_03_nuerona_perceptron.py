from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris_dataset = load_iris()
print(iris_dataset.target_names)

df = pd.DataFrame(np.c_[iris_dataset['data'], iris_dataset['target']], columns= iris_dataset['feature_names'] + ['target'])
print(df.head())


#representación gráfica de los datos
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,7))
plt.scatter(df['petal length (cm)'][df['target']==0], df['petal width (cm)'][df['target']==0], c="b", label="setosa")
plt.scatter(df['petal length (cm)'][df['target']==1], df['petal width (cm)'][df['target']==1], c="r", label="versicolor")

plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.legend(loc="lower right", fontsize=15)

# plt.show()

#representación gráfica de los datos en 3D
from mpl_toolkits import mplot3d
fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.scatter3D(df['petal length (cm)'][df['target']==0], df['petal width (cm)'][df['target']==0], df['sepal width (cm)'][df['target']==0], c="b", label="setosa")
ax.scatter3D(df['petal length (cm)'][df['target']==1], df['petal width (cm)'][df['target']==1], df['sepal width (cm)'][df['target']==1], c="r", label="versicolor")
ax.scatter3D(df['petal length (cm)'][df['target']==2], df['petal width (cm)'][df['target']==2], df['sepal width (cm)'][df['target']==2], c="g", label="virginica")
ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.set_zlabel("sepal width (cm)")
ax.legend(loc="lower right", fontsize=15)
# plt.show()

#entrenamiento del algoritmo perceptron
#reducimos datos a 2 caracterisiticas y 2 clases
df_reduced = df[['petal length (cm)', 'petal width (cm)','target']]
df_reduced = df_reduced[df_reduced['target'] != 2]
X_df = df_reduced[['petal length (cm)', 'petal width (cm)']]
y_df = df_reduced['target']

from sklearn.linear_model import Perceptron

clf = Perceptron(max_iter=1000,random_state=40)
clf.fit(X=X_df,y=y_df)

#mostramos parametros del modelo
print('La formula del modelo es: hw(x) = w1*x1 + w2*x2 + b')
print(f'clf.coef_ que son w1 y w2: {clf.coef_}')
print(f'clf.intercept_ que es b: {clf.intercept_}')

#Representación grafica del limite de decisión
X = X_df.values
mins = X.min(axis=0) - 0.1
maxs = X.max(axis=0) + 0.1
xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 1000), np.linspace(mins[1], maxs[1], 1000))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(10,7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="Set3")
plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), colors='k', linewidths=1)

plt.plot(X[:,0][y_df==0], X[:,1][y_df==0], 'bo', label='setosa')
plt.plot(X[:,0][y_df==1], X[:,1][y_df==1], 'ro', label='versicolor')
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.legend(loc="lower right", fontsize=15)
plt.show()
pass