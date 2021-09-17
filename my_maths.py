from operator import itemgetter
from matplotlib import pyplot as plt

import numpy as np
import math

def k_temp_init(K, T_k, L_0, R_min, graph):
    u = k_rand_paths(graph, K)
    R_a = 0
    beta = 1.5
    while R_a < R_min:
        u, R_a = markov(K, L_0, u, graph, T_k)
        T_k *= beta
    return T_k

def k_rand_paths(graph, k):
    temp_graph = graph.copy()
    np.random.shuffle(temp_graph)
    sublen = int(len(temp_graph) / k)
    if sublen < 3:
        raise Exception("Error: al menos 2 ciudades por viajero.")
    for i in range(k - 1):
        temp_graph.insert((sublen * (i + 1)), -1)
    return temp_graph

def k_mutate(path):
    temp_path = path.copy()
    index = np.random.randint(0, len(temp_path) - 2)
    temp = temp_path[index]
    temp_path[index] = temp_path[index + 1]
    temp_path[index + 1] = temp
    needs_correction, cities_per_traveler = count_cities(temp_path)
    #print(temp_path)
    if needs_correction:
        correct_paths(temp_path, cities_per_traveler)

    return temp_path

def count_cities(path):
    cities = 0
    index = 0
    cities_per_traveler = []
    needs_correction = False
    for city in path:
        if city != -1:
            cities += 1
        else:
            cities_per_traveler.append((index, cities))
            if cities < 3:
                needs_correction = True
            cities = 0
        index += 1
    cities_per_traveler.append((index, cities))
    if cities < 3:
        needs_correction = True

    return needs_correction, cities_per_traveler

def correct_paths(paths, cities_per_traveler):
    needs_correction = True
    while(needs_correction):
        max_cities = max(cities_per_traveler, key=lambda x: x[1])
        min_cities = min(cities_per_traveler, key=lambda x: x[1])

        max_index = max_cities[0]
        min_index = min_cities[0]
        
        gift = paths.pop(max_index - 1)
        paths.insert(min_index - 1, gift)

        needs_correction, cities_per_traveler = count_cities(paths)

def markov(K, L_k, u, graph, T_k):
    j = 0
    for l in range(L_k):
        v = k_mutate(u)
        E_u = sum(k_cost(K, graph, u))
        E_v = sum(k_cost(K, graph, v))
        if E_v <= E_u:
            u = v.copy()
            j += 1
        elif np.random.uniform(0.0, 1.0) < risk(E_u, E_v, T_k):
            u = v.copy()
            j += 1

    return u, j/L_k

def risk(E_u, E_v, T_k):
    n = -(E_v - E_u)/(T_k)
    return math.e ** n

def graph_gen(n):
    nodes = []
    for i in range(n):
        nodes.append((np.random.uniform(0, 1), np.random.uniform(0, 1)))
    return nodes

def euclid_dist(node1, node2):
    npnode1, npnode2 = np.array(node1), np.array(node2)
    dist = np.linalg.norm(npnode1 - npnode2)
    #print(dist)
    return dist

def k_cost(K, graph, path):
    first_city = 0
    index = 0
    cost_list = [0 for city in range(K)]
    for i in range(1, len(path)):
        if path[i] != -1:
            cost_list[index] += euclid_dist(path[i-1], path[i])
        else:
            cost_list[index] += euclid_dist(path[-1], path[first_city])
            first_city = i + 1
            index += 1
            
    return cost_list

def k_annealing(K, T_0, L_k, R_min, graph, iter, alpha):
    energy = []
    best_paths = []
    T_k = k_temp_init(K, T_0, L_k, R_min, graph)
    print('T0 =', T_k)
    k = 0
    u = k_rand_paths(graph, K)
    energy.append(sum(k_cost(K, graph, u)))
    print(k_cost(K, graph, u))
    
    best_paths.append(u)
    while(k < iter - 1):
        u, R_a = markov(K, L_k, u, graph, T_k)
        k += 1
        T_k *= alpha
        if sum(k_cost(K, graph, u)) < sum(k_cost(K, graph, best_paths[-1])):
            best_paths.append(u)
            energy.append(sum(k_cost(K, graph, u)))
        else:
            best_paths.append(best_paths[-1])
            energy.append(sum(k_cost(K, graph, best_paths[-1])))
    print('Eu =', k_cost(K, graph, u), '\tTk =', T_k, '\tRa =', R_a)
    #plt.plot(energy)
    #plt.show()
    return best_paths[-1], best_paths, energy