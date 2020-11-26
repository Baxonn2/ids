from pre_proceso import load_data
from qpso import config_swarm, qpso
import numpy as np

xe, ye = load_data("Data/KDDTrain+_20Percent.txt")

# Agregando unidad de bias (Para training)
N, D = xe.shape
Xe = np.concatenate((xe, np.ones((N, 1))), axis=1)

# Para el testing
#M, D = xv.shape
#Xv = np.concatenate((xe, np.ones((M, 1))), axis=1)

# Cargando archivos de configuracion
# TODO: hacer la funcion de carga de parametros

# Corriendo entrenamiento
X = config_swarm(xe)
w1, w2, MSE = qpso(Xe, ye, X)

print(w1, w2, MSE)
#save('peso', w1, w2, MSE)