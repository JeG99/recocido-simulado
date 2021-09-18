import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from opti_utils import ackley, update_vel

df_experiments = pd.DataFrame()

def poner_menor(row):
    df.at[row.name, 'menor'] = min(row['evaluacion'], df.iloc[row.name - 1].menor)
    return None

# Globals
particle_amount = 5
dim = 2
alpha = 2
beta = 1
max_vel_magnitude = 2
iters = 100

def pso(iters=iters, plot=True):
    '''
    Algoritmo de enjambre de partículas
    '''
    # Inicializa partículas y su velocodad
    particles_pos = np.random.uniform(-32.768, 32.768, size=(particle_amount, dim))
    particles_loc = np.array(particles_pos, copy=True)
    particles_vel = np.zeros_like(particles_pos)

    particles_eval = np.empty(shape=(particle_amount, 2))
    particles_eval[:,0] = list(map(ackley, particles_pos))
    particles_eval[:,1] = particles_eval[:,0]

    # Encuentra minimo global para el inicio del problema
    index_global_min = np.argmin(particles_eval[:,1])
    global_min_pos = particles_pos[index_global_min]
    global_min_eval = particles_eval[:,1][index_global_min]
     
    t = 0
    while t < iters:
        if plot:
            plt.close()
            plt.scatter(particles_pos[:, 0], particles_pos[:, 1])
            plt.scatter(particles_loc[:, 0], particles_loc[:, 1])
            plt.scatter(global_min_pos[0], global_min_pos[1])
            plt.show()

        print("Mejor en iteración " + str(t) +  " " +  str(global_min_pos) )

        # Calcula velocidad de cada partícula
        update_vel(particles_vel, alpha, particle_amount, dim, global_min_pos, particles_pos, particles_loc)
        norms = np.array(particles_pos, copy=True)
        norms[:, 0] = np.linalg.norm(particles_vel, axis=1)
        norms[:, 1] = np.linalg.norm(particles_vel, axis=1)
        # norms = np.linalg.norm(particles_vel, axis=1)
        excedent_vel = np.argwhere(norms > max_vel_magnitude)
        #print(particles_vel[excedent_vel])
        #print(norms[excedent_vel])

        particles_vel[excedent_vel] = max_vel_magnitude * np.divide(particles_vel[excedent_vel], norms[excedent_vel])

        # Calcula la posicion de cada particula
        particles_pos += particles_vel
        particles_pos = np.clip(particles_pos, -32.768, 32.768)
        particles_eval[:,0] = list(map(ackley, particles_pos))
       
        index_global_min = np.argmin(particles_eval, axis=1)
        print(index_global_min)
        particles_eval[:,1] = particles_eval[np.arange(len(particles_eval)), index_global_min] # No entiendo la linea 53 del profe
        index_global_min = np.where(index_global_min == 0)
        particles_loc[index_global_min] = particles_pos[index_global_min]
        index_global_min = np.argmin(particles_eval[:,1])
        global_min_pos = particles_loc[index_global_min,:]
        global_min_eval = particles_eval[index_global_min,1]

        print("Mejor despues de cáclulos " + str(global_min_pos) + " eval: " + str(global_min_eval))
        t += 1

    return global_min_pos 
        

pso()