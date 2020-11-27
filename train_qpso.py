from pre_proceso import load_data
from qpso import config_swarm, qpso
import numpy as np
import random
import configs
from activation import activation
from metrica import metrica

xe, ye = load_data("Data/KDDTrain+_20Percent.txt")

# Agregando unidad de bias (Para training)
N, D = xe.shape
Xe = np.concatenate((xe, np.ones((N, 1))), axis=1)

# Para el testing
#M, D = xv.shape
#Xv = np.concatenate((xe, np.ones((M, 1))), axis=1)

# Cargando archivos de configuracion
# TODO: hacer la funcion de carga de parametros

random.seed(configs.seed)

# Corriendo entrenamiento
X, pBest, pFitness, gBest, gFitness, wBest, Alfa = config_swarm(xe)
w1, w2, MSE = qpso(Xe, ye, X, pBest, pFitness, gBest, gFitness, wBest, Alfa)

print("Resultado")
#print("w1", w1)
#print("w2", w2)
for mse in MSE:
	print(mse)

#save('peso', w1, w2, MSE)

# Testing
xv, yv = load_data("Data/KDDTest+.txt")
wh = np.reshape(w1[:-10], (configs.Nh, D))
H = activation(xv, np.transpose(wh))
zv = w2.dot(np.transpose(H))

accuracy, fscore = metrica(zv, yv)

# TODO: intentar mejorar