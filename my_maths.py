import numpy as np
import math

from matplotlib import pyplot as plt

visits_table = {}

def temp_init(T_k, L_0, R_min, graph):
    u = rand_path(graph)
    R_a = 0
    beta = 1.5
    while R_a < R_min:
        u, R_a = markov(L_0, u, graph, T_k)
        T_k *= beta
    return T_k

def markov(L_k, u, graph, T_k):
    j = 0
    for l in range(L_k):
        v = mutate(u)
        E_u = cost(graph, u)
        E_v = cost(graph, v)
        if E_v <= E_u:
            #print('Ev < Eu')
            u = v.copy()
            j += 1
        elif np.random.uniform(0.0, 1.0) < boltzmann(E_u, E_v, T_k):
            #print('metropolis', boltzmann(E_u, E_v, T_k))
            u = v.copy()
            j += 1

    return u, j/L_k

def boltzmann(E_u, E_v, T_k):
    n = -(E_v - E_u)/(T_k)
    return math.e ** n

def graph_gen(n):
    nodes = []
    for i in range(n):
        nodes.append((np.random.uniform(0, 1), np.random.uniform(0, 1)))
    for city in nodes:
        visits_table[city] = False
    return nodes

def euclid_dist(node1, node2):
    npnode1, npnode2 = np.array(node1), np.array(node2)
    dist = np.linalg.norm(npnode1 - npnode2)
    #print(dist)
    return dist

def cost(graph, path):
    total = 0
    for i in range(1, len(path)):
        total += euclid_dist(path[i-1], path[i])
    total += euclid_dist(path[-1], path[0])
    return total

def rand_path(graph):
    temp_graph = graph.copy()
    np.random.shuffle(temp_graph)
    return temp_graph

def mutate(path):
    temp_path = path.copy()
    index = np.random.randint(0, len(temp_path) - 2)
    temp = temp_path[index]
    temp_path[index] = temp_path[index + 1]
    temp_path[index + 1] = temp
    return temp_path

def visit(node):
    visits_table[node] = True

def annealing(T_0, L_k, R_min, graph, iter, alpha):
    energy = []
    best_paths = []
    T_k = temp_init(T_0, L_k, R_min, graph)
    print('T0 =', T_k)
    k = 0
    u = rand_path(graph)
    energy.append(cost(graph, u))
    best_paths.append(u)
    while(k < iter):
        u, R_a = markov(L_k, u, graph, T_k)
        k += 1
        T_k *= alpha
        if cost(graph, u) < cost(graph, best_paths[-1]):
            best_paths.append(u)
            energy.append(cost(graph, u))
        #print('Eu =', cost(graph, u),'\t\tTk =', T_k, '\t\tRa =', R_a)
    print('Eu =', cost(graph, u),'\t\tTk =', T_k, '\t\tRa =', R_a)
    plt.plot(energy)
    plt.show()
    return best_paths[-1]