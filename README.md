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
