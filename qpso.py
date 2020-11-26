from rand_W import rand_W
import numpy as np
from math import inf, log
import configs
from random import random
from activation import activation

def ini_swarm(Np, Nh, D):
    X = np.zeros((Np, Nh*(D)))
    for i in range(Np):
        wh = rand_W(Nh, D)
        X[i:] = np.reshape(wh, (1, Nh*D)) #? Que es esto Puede que falta una coma
    return X

def config_swarm(xe):
    #* Esto es temporal
    Np      = configs.Np
    Nh      = configs.Nh
    MaxIter = configs.MaxIter

    N, D = xe.shape
    D += 1
    X = ini_swarm(Np, Nh, D)
    N, Dim = X.shape
    pBest = np.zeros((Np, Dim))
    pFitness = np.ones((1, Np)) * inf
    gBest = np.zeros((1, Dim))
    gFitness = inf
    wBest = np.zeros((1, Nh))

    Alfa = (0.95 - 0.2) * (MaxIter - np.array(range(1, MaxIter + 1))) \
            / MaxIter + 0.2 # Cofft

    return X, pBest, pFitness, gBest, gFitness, wBest, Alfa

def mlp_pinv(H, ye, C):
    N, L = H.shape
    H_ = np.transpose(H)
    yh = ye.dot(H)
    hh = (H_.dot(H) + np.eye(L) / C)
    w2 = yh.dot(np.linalg.pinv(hh))

    return w2

def mse(ye, ze):
    return ((ye - ze)**2).mean()

def fitness(Xe, ye, Nh, X, Cpinv):
    N, D = Xe.shape
    Np = configs.Np

    # Shape (Np)
    W2 =  np.zeros((Np, Nh)) # TODO: inicializar los pesos

    # Posiblemente dejemos la cagá
    MSE = np.full(Np, -1.0)

    for i in range(Np):
        p = X[i,:]
        w1 = np.reshape(p, (D, Nh))
        H = activation(Xe, w1)
        W2[i,:] = mlp_pinv(H, ye, configs.C)
        ze = W2[i,:].dot(np.transpose(H))

        MSE[i] = np.sqrt(mse(ye, ze)) # Error cuadratico medio

    # FIXME: esto está mal porque da cero siempre
    return MSE, W2

def upd_particle(X, pBest, pFitness, gBest, gFitness, new_pFitness, new_beta, wBest):
    idx = np.where(new_pFitness < pFitness) # FIXME: arreglar esto... hacer funcion find?
    
    print("new_pFitness", new_pFitness, "\npFitness", pFitness)
    print("idx[0].size", idx[0].size)
    if idx[0].size > 0: # Si es que se encontró una particula mejor
        for i, j in zip(idx[0], idx[1]):
            pFitness[i,j] = new_pFitness[j]
            print("pBest.shape", pBest.shape)
            print("X.shape", X.shape)
            pBest[j,:] = X[j,:]

    new_gFitness, idx = min_(pFitness)  # TODO: programar esta funcion para encontrar el menor Fitnees (Retorna su valor y el indice)

    if new_gFitness < gFitness:
        gFitness = new_gFitness
        gBest = pBest[idx, :]
        wBest = newBeta[idx, :]

    return pBest, pFitness, gBest, gFitness, wBest

def qpso(Xe, ye, X, pBest, pFitness, gBest, gFitness, wBest, Alfa):
    Np      = configs.Np
    Nh      = configs.Nh
    MaxIter = configs.MaxIter

    # Inicializando MSE
    MSE = np.full(MaxIter, -1)

    N, D = Xe.shape

    for iter_ in range(MaxIter):
        new_pFitness, new_beta = fitness(Xe, ye, Nh, X, configs.C)
        pBest, pFitness, gBest, gFitness, wBest = upd_particle(X, pBest,
            pFitness, gBest, gFitness, new_pFitness, new_beta, wBest)

        MSE[iter_] = gFitness
        mBest = mean_column(pBest)      # TODO: Programar el promedio de las columnas
        for i in range(Np):
            for j in range(Dim):
                phi = random()
                u = random()
                pBest[i,j] = phi * pBest[i, j] + (1 - phi) * gBest[j]
                if random() > 0.5:
                    X[i,j] = pBest[i, j] + Alfa[iter_] * abs(mBest[j] - X[i,j]) * log(1/u)
                else:
                    X[i,j] = pBest[i, j] - Alfa[iter_] * abs(mBest[j] - X[i,j]) * log(1/u)

    return (gBest, wBest, MSE)






