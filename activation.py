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
    print("[activation] z.shape", z.shape)
    H = 1./(1+exp(-z))
    print("[activation] H.shape", H.shape)
    return H

def activationz(z):
    H = 1./(1+exp(-z))
    return H