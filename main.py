from my_maths import *
from matplotlib import pyplot as plt
import itertools

def run_experiment(k):
    graph = graph_gen(10)
    path, best_paths = k_annealing(k, 0.001, 100, 0.95, graph, 1000, 1.00001)

    paths = [[] for t in range(k)]
    traveler = 0
    for city in path:
        if city != -1:
            paths[traveler].append(city)
        else:
            traveler += 1        

    for j in paths:
        x_seq = [city[0] for city in j]
        x_seq.append(j[0][0])
        y_seq = [city[1] for city in j]
        y_seq.append(j[0][1])
        plt.scatter(x_seq, y_seq)
        plt.plot(x_seq, y_seq)
    plt.show()

run_experiment(3)