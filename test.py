from forward import forward

# TODO: esto debería cargar la data
Xv, Yv = load_testing_data() #* Xv son los valores de entrada y el Yv son las etiquetas

# TODO: esto debería preparar los pesos
w1, bias, w2 = load_pesos()

# Calculando data de salida del IDS
z = forward(Xv, w1, bias, w2)

# Obteniendo metricas
accuracy, fscore = metrica(z, Yv)

# Guardando las metricas resultantes
# TODO: Esto debería guardar metrica.csv
savetxt('metrica.csv', accuracy, fscore) 
# El formato del archivo tiene que ser:
# accuracy, fscore_clase_1, fscore_clase_2