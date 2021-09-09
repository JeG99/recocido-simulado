import numpy as np
from matplotlib import pyplot as plt

visits_table = {}
T_0 = 0.1
def temp_init(T_k, L_0, R_min, graph):
    u = rand_path(graph)
    R_a = 0
    beta = 1.5
    while R_a < R_min:
        u, R_a = markov(L_0, u, graph, T_k)
        T_k *= beta
    return T_k

def markov(L_k, u, graph, T_k=T_0):
    j = 0
    for l in range(L_k):
        v = mutate(u)
        E_u = cost(graph, u)
        E_v = cost(graph, v)
        if E_v <= E_u:
            #print(E_v,'less or equal than', E_u)
            u = v
            j += 1
        elif np.random.uniform(0.0, 1.0) < metropolis(E_u, E_v, T_k):
            print('metropolis')
            u = v
            j += 1
        else:
            print('no')
    
    
#    x_seq = [city[0] for city in u]
#    x_seq.append(u[0][0])
#    y_seq = [city[1] for city in u]
#    y_seq.append(u[0][1])
#    plt.scatter(x_seq, y_seq)
#    plt.plot(x_seq, y_seq)
#    plt.savefig("hola")
#    plt.show()


    return u, j/L_k
    
def metropolis(E_u, E_v, T_k):
    a = -(E_v - E_u)/(T_k)
    return np.exp(a)

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
    temp_graph = graph
    np.random.shuffle(temp_graph)
    return temp_graph

def mutate(path):
    temp_path = path
    index = np.random.randint(0, len(temp_path) - 2)
    temp = temp_path[index]
    temp_path[index] = temp_path[index + 1]
    temp_path[index + 1] = temp
    return temp_path

def visit(node):
    visits_table[node] = True

def annealing(T_0, L_k, R_min, graph, iter, alpha):
    T_k = temp_init(T_0, L_k, R_min, graph)
    k = 0
    u = rand_path(graph)
    while(k < iter):
        u, R_a = markov(L_k, u, graph, T_k)
        k += 1
        T_k *= alpha
    return u