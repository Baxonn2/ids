from activation import activation
from metrica import metrica
from pre_proceso import load_data
import numpy as np
from configs import *

def load_w(file_path="pesos.npy"):
    return np.load(file_path, allow_pickle=True)

def save(accuracy, fscore):
    with open("metricas.csv", "w") as f:
        f.write(f"{accuracy},{fscore[0]},{fscore[1]}")

def test(file_path):
    # Cargando data
    data_test = "./csv_files/KDDTest.txt"
    xv, yv = load_data(data_test)
    Xv = np.vstack([xv, np.full((1, xv.shape[1]), 1)])

    w1, w2 = load_w()

    D = Xv.shape[0]
    wh = np.reshape(w1, (Nh, D))
    H = activation(wh, Xv)
    zv = np.matmul(w2, H)

    accuracy, fscore = metrica(zv, yv)

    save(accuracy, fscore)

if __name__ == "__main__":
    test('./csv_files/KDDTest+.txt')