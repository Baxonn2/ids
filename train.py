import numpy as np
from rand_W import rand_W
from activation import activationz
from pre_proceso import load_data

def upd_date(xe, ye, L, C):
    shape = xe.shape()
    w1 = rand_W(L, shape[0])
    bias = rand_W(L, 1)
    bias_matrix = np.reshape(bias, (1, shape[1]))

    z = w1 * xe + bias_matrix
    H = activationz(z)

    # Calculando pesos de salida
    Hp = np.transpose(H)            
    yh = ye * Hp
    hh = (H * Hp + np.eye(L) / C)    
    inv = np.linalg.pinv(hh)         
    w2 = yh * inv

    return w1, bias, w2


# TODO: programar la data y los parametros de configuracion
Xe, Ye = load_data("Data/KDDTrain+_20Percent.txt")
print("Xe:")
print(Xe.head())
print("\nYe:")
print(Ye.head())

param_config = "Parametros de configuracion"  # TODO: programar config
L = 20    # Nodos en la capa oculta  (Cambiar por param_config[L])
C = 1     # Penalidad pinv (Cambiar por param_config[C])

w1, bias, w2 = upd_date(Xe, Ye, L, C)
#costos = calc_costo(Xe, Ye, w1, bias, w2)   # TODO: descomentar esto y programar esta funcion

# Grabando pesos del IDS
#* Este archivo se puede llamar como queramos
save('pesos', w1, bias, w2)   # TODO: esto debería guardar los pesos en binario

# Grabando vector de costo
savetxt('costo.csv', costo)   # TODO: esto debería guardar el csv costo