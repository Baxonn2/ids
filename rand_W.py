from math import sqrt
import numpy as np

def rand_W(next_nodes, current_nodes):
    r = sqrt(6 / (next_nodes + current_nodes))
    w = rand(next_nodes, current_nodes) * 2 * r - r;
    return w

def rand(next_nodes, current_nodes):
    return np.random.rand(next_nodes, current_nodes)