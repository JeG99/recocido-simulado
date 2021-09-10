from my_maths import *
from matplotlib import pyplot as plt

def run_experiment():
    graph = graph_gen(6)
    path = annealing(0.3, 100, 0.9, graph, 500, 1.1)
    x_seq = [city[0] for city in path]
    x_seq.append(path[0][0])
    y_seq = [city[1] for city in path]
    y_seq.append(path[0][1])
    plt.scatter(x_seq, y_seq)
    plt.plot(x_seq, y_seq)
    plt.show()

run_experiment()