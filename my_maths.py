import numpy as np

visits_table = {}

def temp_init(T_k, L_0, R_min, graph):
    u = rand_path(graph)
    R_a = 0
    beta = 1.5
    while R_a < R_min:
        u, R_a = markov(u, L_0)
        T_k *= beta
    return T_k    
    
def markov():
    

def graph_gen(n):
    nodes = []
    for i in range(n):
        nodes.append(np.array([np.random.random_sample(size=1), np.random.random_sample(size=1)]))
    for city in nodes:
        visits_table[city] = False
    return nodes

def euclid_dist(node1, node2):
    dist = numpy.linalg.norm(node1 - node2)

def cost(graph, path):
    total = 0
    for i in range(1, len(path)):
        total += euclid_dist(path[i-1], path[i])
    total += euclid_dist(path[-1], path[0])
    return total

def rand_path(graph):
    path = np.random.shuffle(graph)
    return path

def visit(node):
    visits_table[node] = True