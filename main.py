from my_maths import *
from matplotlib import pyplot as plt
import itertools

def run_experiment():
    graph = graph_gen(12)
    path = k_annealing(2, 0.1, 10, 0.9, graph, 100, 1.1)
    
    #x_seq = [city[0] for city in path]
    #x_seq.append(path[0][0])
    #y_seq = [city[1] for city in path]
    #y_seq.append(path[0][1])
    #plt.scatter(x_seq, y_seq)
    #plt.plot(x_seq, y_seq)
    #plt.show()

    paths = []
    result = [list(v) for k,v in itertools.groupby(path, key= lambda x: x==-1) if not k]

    for j in result:
        x_seq = [city[0] for city in j]
        x_seq.append(j[0][0])
        y_seq = [city[1] for city in j]
        y_seq.append(j[0][1])
        plt.scatter(x_seq, y_seq)
        plt.plot(x_seq, y_seq)
        plt.show()

run_experiment()