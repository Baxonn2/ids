import numpy as np
import pandas as pd

def confusion_matrix(yv, zv):
	real = pd.Series(yv, name="real")
	pred = pd.Series(zv, name="pred")
	return pd.crosstab(real, pred)

def precision(i, cm):
	suma = 0
	for j in (-1, 1):
		suma += cm[i][j]
	return cm[i][i] / suma

def recall(j, cm):
	suma = 0
	for i in (-1, 1):
		suma += cm[i][j]
	return cm[j][j] / suma

def fscore(j, cm):
	return 2 * precision(j, cm) * recall(j, cm) / (precision(j, cm) + recall(j, cm))


def accuracy_fun(zv, yv):
	acc = zv == yv
	return acc.mean()

def metrica(zv, yv):
	zv[zv>=0] = 1
	zv[zv<0] = -1

	cm = confusion_matrix(yv, zv)
	print(cm)

	# Obteniendo fscores
	fscore_result = []
	for j in (1, -1):
		fscore_result.append(fscore(j, cm))
	print("Fscore (%) =", fscore_result)

	accuracy = accuracy_fun(zv, yv)
	print("accuracy", accuracy)

	return accuracy, fscore_result