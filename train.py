import numpy as np
import math
from configs import *
import pre_proceso as pp
import metrica as m

"""
xe: data de entrada
ye: data deseada (salida)
"""

# TODO: crear preprocesador
# Nh: nodos en capa oculta

# def upd_date(xe, ye, L, C):
#     """
#     Etapa de alimentacion unidereccional. AKA forward
#     :param xe: capa de entrada
#     :param ye: datos de salida
#     :param L: neuronas de la capa
#     :param C: Parametro en ranfo de 10^n con n = {1...10}
#     :return:
#     """
#     rows, cols = np.shape(xe)
#     # #Capa entrada x capa oculta
#     # Capa de entrada a capa oculta
#     w1 = rand_W(L, rows)
#     # bias
#     bias = rand_W(L, 1)
#     # Matriz LxN
#     bias_matrix = np.repeat(bias, cols, axis=1)
#
#     # Operacion + bias
#     z = np.matmul(w1, xe) + bias_matrix
#
#     # Resultado de la matriz de activacion
#     H = sigmoid(z)
#
#     # Calcular pesos de salida
#     yh = np.matmul(ye, np.transpose(H))
#     hh = (np.matmul(H, np.transpose(H)) + np.identity(L)/C)
#     # calculo de pseudoinversa
#     pinv = np.linalg.pinv(hh)
#     w2 = np.matmul(yh, pinv)
#
#     return w1, bias, w2

# def save(Xe, Ye, L, C):
#     w1, bias, w2 = upd_date(Xe, Ye, L, C)
#
#     with open('pesos.np', 'wb') as pesos:
#         np.save(w1)
#         np.save(bias)
#         np.save(w2)


def r_function(n_in=41, n_hidden:int = 2):
    return math.sqrt(6/(n_in + n_hidden))


def rand_W(next_nodes, current_nodes):
    """
    Define los pesos de forma aleatoria entre la capa actual y la siguiente
    :param next_nodes:
    :param current_nodes:
    :return:
    """
    r = r_function(next_nodes, current_nodes)
    w = np.random.random((next_nodes, current_nodes)) * 2 * r - r
    return w


def sigmoid(z):
    """
    sigmoid function
    :param z: output of the network
    :return:
    """
    H = 1./(1+np.exp(-z))
    return H


def ini_swarm(Np, Nh, D):
    """
    Crea la poblacion inicial para el algoritmo QPSO a partir de una cantidad de
    particulas y tamano de capas en la red
    :param Np: Cantidad de particulas a crear
    :param Nh: Cantidad de neuronas en la capa oculta
    :param D: Cantidad de atributos de las muestras
    :return: Conjunto de Np particulas ordenadas en un arreglo de Np×(Nh*D)
    """
    X = np.zeros((Np, Nh*D))
    for i in range(Np):
        # creacion de pesos aleatorios capa entrada x capa oculta
        wh = rand_W(Nh, D)
        # aplanado de pesos a 1 dimension para la red_i
        X[i, :] = np.reshape(wh, (1, Nh * D))
    return X


def config_swarm(xe, Np, Nh, MaxIter):
    """

    :param xe: Datos modificados (fila extra)
    :param Np: Numero de particulas
    :param Nh:
    :return:
    """
    # Nueva dimension
    D = xe.shape[0]
    # Particulas
    X = ini_swarm(Np, Nh, D)
    # Largo de los pesos de la capa de entrada a la capa oculta aplanados
    Dim = X.shape[1]

    # Mejor particula encontrada
    pBest = np.zeros((Np, Dim))
    pFitness = np.full((1, Np), 1) * np.inf
    # Global Best
    gBest = np.zeros((1, Dim))
    # Global fitness
    gFitness = np.inf
    # Best weight
    wBest = np.zeros((1, Nh))
    alfa = (0.95 - 0.2) * (MaxIter - np.array(range(1, MaxIter + 1))) \
            / MaxIter + 0.2

    return X, Dim, pBest, pFitness, gBest, gFitness, wBest, alfa


def activation(x, w):
    """
    activacion para qpso
    :param x:
    :param w:
    :return:
    """
    z = np.matmul(w, x)
    H = sigmoid(z)
    return H


def mlp_pinv(H, ye):
    """
    Calcula la pseudo inversa para calcular w2
    :param H: Resultado de funcion de activacion aplicada
    :param ye: Solucion esperada de la prediccion
    :return: Nuevos valores para los pesos w2
    """
    L = H.shape[0]
    yh = np.matmul(ye, np.transpose(H))
    hh = (np.matmul(H, np.transpose(H)) + np.identity(L) / C)
    # calculo de pseudoinversa
    pinv = np.linalg.pinv(hh)
    w2 = np.matmul(yh, pinv)

    return w2


def mse(ye, ze):
    return ((ye - ze) ** 2).mean()


def upd_particle(X, pBest, pFitness, gBest, gFitness, New_pFitness, newBeta, wBest):
    """
    actualizacion de fitness
    :param X: Particulas
    :param pBest: Mejor set de particulas
    :param pFitness: Fitness de las particulas
    :param gBest: Optimo global
    :param gFitness: fitness global
    :param New_pFitness: Nuevo fitness calculado
    :param newBeta: Nuevos pesos de salida calculados
    :param wBest: Mejores pesos de salida encontrados (1 particula)
    :return: Actualizacion de las mejores particulas, fitness y mejor solucion
    encontrada
    """
    idx = np.where(New_pFitness < pFitness)[1]
    if len(idx) > 0:
        pFitness[0][idx] = New_pFitness[idx]
        # Actualiza el mejor estado de las particulas
        pBest[idx, :] = X[idx, :]
    # Index de nuevo global fitness
    idx_gFitness = np.where(pFitness == min(pFitness))[0][0]
    # Actualiza el mejor resultado de la poblacion
    if pFitness[0][idx_gFitness] < gFitness:
        # Mejor fitness obtenido
        gFitness = pFitness[0][idx_gFitness]
        # Mejor particula obtenida (pesos entrada × capa oculta)
        gBest = pBest[idx_gFitness, :]
        # Mejores pesos de salida
        wBest = newBeta[idx_gFitness, :]

    return pBest, pFitness, gBest, gFitness, wBest


def fitness(xe, ye, Nh, X):
    """
    Obtiene los nuevos pesos de salida
    el fitness se define como el MSE. Es funcion de minimizacion
    :param xe: Datos de entrada
    :param ye: Datos de salida
    :param X: Conjunto de particulas
    :param Nh: Nodos en capa oculta
    :return: fitness (MSE) y nuevos pesos de salida encontrados
    """
    # forma de data de entrada
    rows, cols = xe.shape
    # Nro particulas
    Np = X.shape[0]
    # arreglo de pesos w2 para cada particula
    w2 = np.zeros((Np, Nh))
    # arreglo de fitness de cada particula
    MSE = np.full(Np, np.inf)

    # Recorre cada particula
    for i in range(Np):
        p = X[i, :]
        # Convierte los pesos aplanados a su forma real
        w1 = np.reshape(p, (Nh, rows))
        H = activation(xe, w1)
        w2[i, :] = mlp_pinv(H, ye)
        # Obtiene vector ze lista de soluciones
        ze = np.matmul(w2[i, :], H)
        # Error cuadratico medio entre ye y ze obtenido
        MSE[i] = np.sqrt(mse(ye, ze))
    return MSE, w2


def qpso(xe, ye):
    # Np: numero de particulas representa las filas
    # Representa las columnas
    # Mejor MSE de cada iteración
    MSE = np.full(MaxIter, np.inf)

    # Configuracion inicial de la busqueda
    X, Dim, pBest, pFitness, gBest, gFitness, wBest, alfa = config_swarm(xe, Np, Nh, MaxIter)

    # Iteraciones del algoritmo
    for it in range(MaxIter):
        # MSE, w2
        New_pFitness, New_Beta = fitness(xe, ye, Nh, X)

        # Se actualiza y retorna informacion
        # Particulas actualizadas, Fitness de las particulas, Mejor particula,
        # Fintess de mejor particula, Mejores pesos de salida
        pBest, pFitness, gBest, gFitness, wBest = upd_particle(X, pBest,
                                                               pFitness, gBest,
                                                               gFitness, New_pFitness,
                                                               New_Beta, wBest)

        # El fitness en la iteracion corresponde al mejor fitness obtenido

        MSE[it] = gFitness
        print('iteracion:', it, '\tfitness: ', MSE[it])
        # Promedio de los mejores elementos de las particulas
        mBest = np.mean(pBest, axis=1)

        # Recorre la lista de particulas
        for i in range(Np):
            # Recorre cada elemento en una particula
            for j in range(Dim):
                # Valores aleatorios para generar un movimiento
                phi, mu, rand = np.random.random(3)
                # Actualizacion de la particula con respecto al elite
                pBest[i, j] = phi * pBest[i, j] + (1 - phi)*gBest[j]
                # Actualizacion de la lista de particulas X
                if rand > 0.5:
                    X[i, j] = pBest[i, j] + alfa[it] * abs(mBest[i] - X[i, j]) * np.log(1 / mu)
                else:
                    X[i, j] = pBest[i, j] - alfa[it] * abs(mBest[i] - X[i, j]) * np.log(1 / mu)

    return gBest, wBest, MSE


def load_param_config():
    pass


def save(w1, w2, file_path="pesos.npy"):
    with open(file_path, 'wb') as pesos:
        np.save(pesos, np.array([w1, w2], dtype=object))
        #np.save(bias)

def save_mse(mse):
    np.savetxt("costos.csv", mse, delimiter=",", fmt="%f")

if __name__ == '__main__':
    data = "./csv_files/KDDTrain.txt"
    xe, ye = pp.load_data(data)
    
    load_param_config()
    # Agrega fila a datos de entrada
    
    Xe = np.vstack([xe, np.full((1, xe.shape[1]), 1)])

    w1, w2, mse_ = qpso(Xe, ye)

    # Guardando pesos
    save(w1, w2)

    # Colocar aqui la funcion de costos
    save_mse(mse_)
    


