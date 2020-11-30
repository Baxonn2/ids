import pandas as pd
import numpy as np


def load_data(path):
    result = pd.read_csv(path, header=None)
    print(result.shape)
    # print(result.info())
    # result.iloc[:,[1]].astype(str)#.astype(int)

    groups = {
        1: result.iloc[:, 1].unique(),
        2: result.iloc[:, 2].unique(),
        3: result.iloc[:, 3].unique(),
        41: result.iloc[:, 41].unique()
    }

    def to_numbers(key: int, value):
        #  Caso en que se está tratando la salid
        # La salida se trata como valor 1 en caso de que sea "normal" y
        # -1 en cualquier otro caso
        if key == 41:
            if value == "normal":
                return 1
            else:
                return -1
        i, = np.where(groups[key] == value)
        return i[0]

    # Actualizando dataframe
    for k in groups.keys():
        result.iloc[:, k] = result.iloc[:, k].apply(lambda x: to_numbers(k, x))

    # print("\nConvertido")
    # print(result.info())

    Ye = result[41]
    result.drop(41, axis='columns', inplace=True)

    # Eliminando la ultima columna
    result.drop(result.columns[[-1,]], axis=1, inplace=True) 
    
    return result.transpose(), Ye


if __name__ == '__main__':
    Xe, Ye = load_data('./csv_files/KDDTrain+sm.txt')
    print(Xe.shape)

    print(Ye.shape)
