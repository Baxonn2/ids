from pre_proceso import load_data
from qpso import config_swarm, qpso
import numpy as np

xe, ye = load_data("Data/KDDTrain_chica.txt")

# Agregando unidad de bias (Para training)
N, D = xe.shape
Xe = np.concatenate((xe, np.ones((N, 1))), axis=1)

# Para el testing
#M, D = xv.shape
#Xv = np.concatenate((xe, np.ones((M, 1))), axis=1)

# Cargando archivos de configuracion
# TODO: hacer la funcion de carga de parametros

# Corriendo entrenamiento
X, pBest, pFitness, gBest, gFitness, wBest, Alfa = config_swarm(xe)
w1, w2, MSE = qpso(Xe, ye, X, pBest, pFitness, gBest, gFitness, wBest, Alfa)

print(w1, w2, MSE)
#save('peso', w1, w2, MSE)