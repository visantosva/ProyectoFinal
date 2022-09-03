# ProyectoFinal
Participantes en el proyecto de la materia de tratamiento de datos son: Alvaro Misael Criollo Rodas, David Octavio Briones Rodriguez, Victor Avelino Santos Valarezo,  Karla Jacqueline Calapi Mena
# Set de datos: MNIST
#
# Contiene 60,000 datos de entrenamiento y 10,000 de validación. 
# Cada imagen es de 28x28 pixeles. La clasificación se llevará a cabo
# usando una red neuronal con una capa oculta que contiene 15 neuronas.
# 
# codificandobits.com - 2018

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils

import matplotlib.pyplot as plt
import numpy as np

#
# Lectura, visualización y pre-procesamiento de los datos
#

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualizaremos 16 imágenes aleatorias tomadas del set x_train
ids_imgs = np.random.randint(0,x_train.shape[0],16)
for i in range(len(ids_imgs)):
	img = x_train[ids_imgs[i],:,:]
	plt.subplot(4,4,i+1)
	plt.imshow(img, cmap='gray')
	plt.axis('off')
	plt.title(y_train[ids_imgs[i]])
plt.suptitle('16 imágenes del set MNIST')
plt.show()

# Pre-procesamiento: para introducirlas a la red neuronal debemos
# "aplanar" cada una de las imágenes en un vector de 28x28 = 784 valores

X_train = np.reshape( x_train, (x_train.shape[0],x_train.shape[1]*x_train.shape[2]) )
X_test = np.reshape( x_test, (x_test.shape[0],x_test.shape[1]*x_test.shape[2]) )

# Adicionalmente se normalizarán las intensidades al rango 0-1
X_train = X_train/255.0
X_test = X_test/255.0

# Finalmente, convertimos y_train y y_test a representación "one-hot"
nclasses = 10
Y_train = np_utils.to_categorical(y_train,nclasses)
Y_test = np_utils.to_categorical(y_test,nclasses)
