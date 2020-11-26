from numpy import exp

def activation(x, w):
    """
    Funcion sigmoid de activacion

    Args:
        x : Valor del nodo anterior
        w : Peso

    Returns:
        retorna el valor de salida del nodo o neurona
    """
    z = x.dot(w);
    H = 1./(1+exp(-z))
    return H

def activationz(z):
    H = 1./(1+exp(-z))
    return H