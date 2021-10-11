from typing import no_type_check
from deap import base, creator, algorithms, tools, gp
import operator
import random
import numpy as np
import math
from sklearn.metrics import mean_absolute_error

MAX_WEIGHT = 6404180
CROSS_PROB = 0.7
MUT_PROB = 0.8
N_GEN = 1000
INIT_SIZE = 100
#best profit: 13549094

toolbox = base.Toolbox()
with open('p08_p.txt', 'r') as file:
    profit = file.readlines()

with open('p08_w.txt', 'r') as file:
    weight = file.readlines()

creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)

items = [(int(profit[i]), int(weight[i])) for i in range(0, len(profit))]

toolbox.register("attr_item", random.randrange, len(profit))
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_item, INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    weight = 0
    value = 0
    for i in individual:
        weight += items[i][0]
        value += items[i][1]
    if weight > MAX_WEIGHT:
        return 1000, 0  # castigo por si se pasa del peso
    return weight, value


def mateMethod(ind1, ind2):
    temp = set(ind1)
    ind1 &= ind2
    ind2 ^= temp
    return ind1, ind2


def mutateMethod(individual):
    if random.random() < 0.5:
        if len(individual) > 0:
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(len(profit)))
    return individual,
    
toolbox.register("evaluate", evaluate)
toolbox.register("mate", mateMethod)
toolbox.register("mutate", mutateMethod)
toolbox.register('select', tools.selNSGA2)

hof = tools.ParetoFront()
pop = toolbox.population(n=100)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis = 0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

results, log = algorithms.eaSimple(
    population=pop, toolbox=toolbox, cxpb=CROSS_PROB, mutpb=MUT_PROB, ngen=N_GEN, stats=stats, halloffame=hof, verbose=True)

print(results)
print(log)

