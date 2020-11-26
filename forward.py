import numpy as np
from activation import activationz

def forward(Xv, w1, bias, w2):
    shape = Xv.shape()
    bias_matrix = np.reshape(bias, (1, shape[1]))
    z = w1 * Xv + bias_matrix
    H = activationz(z)
    z = w2 * H

    return z