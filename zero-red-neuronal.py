import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# se crea el dataset
n=500
p=2 #caracteristicas de cada uno de los registros

x, y = make_circles(n_samples=n, factor=0.5, noise=0.05) #el vector y es binario, tiene 0 o 1 según sea el grupo al que pertenece
y = y[:, np.newaxis]

plt.scatter(x[y[:, 0] ==0, 0], x[y[:, 0] ==0,1], c="salmon") #todos los puntos en el eje x y la columna 1, y en el eje y, todos los puntos y la columna 2
plt.scatter(x[y[:, 0] ==1, 0], x[y[:, 0] ==1,1])
plt.axis("equal")
plt.show()

class capa():
  def __init__(self, conexiones, neuronas, activacion):
    self.act = activacion
    self.b = np.random.rand(1, neuronas) * 2 -1 #bias
    self.w =  np.random.rand(conexiones, neuronas) * 2 -1 #peso

#funciones de activacion

#funcion sigmoide (lambda es funcion anonima)
sigmoide = (lambda x: 1 / (1+np.e **(-x)),
            lambda x: x*(1-x))

relu = lambda x: np.maximum(0, x)

_x = np.linspace(-5, 5, 100) #vector de valores que van de -5 a 5 de forma lineal y genera 100 valores
plt.plot(_x, sigmoide[1](_x)) #se muestra el indice 0 para ver la funcion, el indice 1 es para ver su derivada
#plt.plot(_x, relu(_x))

lo = capa(p, 4, sigmoide)
l1 = capa(4, 8, sigmoide)

def crearNn(topology, activacion):
  nn=[]
  for l, layer in enumerate(topology[:-1]):
    nn.append(capa(topology[l], topology[l+1], activacion))
  return nn

topology = [p,4,8,16,8,4,1]

neuralNet = crearNn(topology, sigmoide)

l2Cost = (lambda yPredict, yReal: np.mean((yPredict - yReal) ** 2),
          lambda yPredict, yReal: (yPredict - yReal))

def train(neuralNet, x, y, l2Cost, learningR = 0.5, train = True): #lr es un valor por el que se multiplica el valor gradiente, nos permite saber en que grado estamos actualizando los parametros

  out = [(None, x)]
  for l, layer in enumerate(neuralNet):
    z = out[-1][1] @ neuralNet[l].w + neuralNet[l].b # @ multipliacion matricial
    a = neuralNet[l].act[0](z)
    out.append((z,a))
  print(l2Cost[0](out[-1][1], y))

  if train:
    #train : backpropagation y gradient descent
    deltas = []

    for l in reversed(range(0, len(neuralNet))):
      z = out[l+1][0]
      z = out[l+1][1]
      if(l == len(neuralNet)-1):
        #calcular ultima capa
        deltas.insert(0, l2Cost[1](a,y) * neuralNet[l].act[1](a))
      else:
        #calcular delta respecto a capa previa
        deltas.insert(0, deltas[0] @ _w.T * neuralNet[l].act[1](a))

      _w = neuralNet[l].w

      neuralNet[l].b = neuralNet[l].b - np.mean(deltas[0], axis=0, keepdims=True)*learningR
      neuralNet[l].w = neuralNet[l].w - out[l][1].T @ deltas[0] * learningR

  return out[-1][1]

train(neuralNet, x, y, l2Cost, 0.5)