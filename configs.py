import pandas as pd

# Cargando configuraciones
configs = pd.read_csv("configs.csv")

# Iteraciones de QPSO
MaxIter = configs["MaxIter"][0]
# Numero de particulas
Np = configs["Np"][0]
# Numero de neuronas en la capa oculta
Nh = configs["Nh"][0]
# Parametro C
C = configs["C"][0]
