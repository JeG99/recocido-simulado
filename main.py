from my_maths import *
from matplotlib import pyplot as plt

import itertools
import pandas as pd

def poner_menor(row):
    df.at[row.name, 'menor'] = min(row['evaluacion'], df.iloc[row.name - 1].menor)
    return None

df_experiments = pd.DataFrame()

k = 3
T0 = 0.1
Lk = 70
Rmin = 0.95
graph = graph_gen(10)
iterations = 500
alpha = 1.00001

n_experiments = 10

figure, axis = plt.subplots(2, 5)

def run_experiment(experiment_number, max_experiments):
    path, best_paths, cost_sum = k_annealing(k, T0, Lk, Rmin, graph, iterations, alpha)

    paths = [[] for t in range(k)]
    traveler = 0
    for city in path:
        if city != -1:
            paths[traveler].append(city)
        else:
            traveler += 1        

    row = int((experiment_number - 1) / int(max_experiments / 2))
    col = int(experiment_number) - (row * int(max_experiments / 2) + 1)
    for j in paths:
        x_seq = [city[0] for city in j]
        x_seq.append(j[0][0])
        y_seq = [city[1] for city in j]
        y_seq.append(j[0][1])  
        axis[row, col].scatter(x_seq, y_seq)
        axis[row, col].plot(x_seq, y_seq)
        axis[row, col].set_title("Experimento " + str(experiment_number))

    return best_paths, cost_sum

exp_n = 1
for i in range(n_experiments):
    paths, cost = run_experiment(exp_n, n_experiments)
    exp_n += 1

    cantidad = len(paths)
    df = pd.DataFrame(
        {'algoritmo':["Annealing"] * cantidad,
        'experimento':[i]*cantidad,
        'iteracion':list(range(0, cantidad)),
        'path':list(paths),
        'evaluacion':list(cost)}
    )
    print(df)
    df_experiments = df_experiments.append(df)

plt.show()
plt.close()

df_experiments.reset_index(drop=True, inplace=True)
results = df_experiments.groupby('iteracion').agg({'evaluacion': ['mean', 'std']})
print(results)

promedios = results['evaluacion']['mean'].values
std = results['evaluacion']['std'].values
plt.plot(range(0,cantidad), promedios, color='red', marker='*')
plt.plot(range(0,cantidad), promedios+std, color='b', linestyle='-.')
plt.plot(range(0,cantidad), promedios-std, color='b', marker='o')
plt.xlabel('iteraciones')
plt.ylabel('menor encontrado')
plt.legend(['promedio', 'promedio+std','promedio-std'])
plt.title('Recocido simulado')
plt.show()