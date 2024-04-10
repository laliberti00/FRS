import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

# Percorso alla cartella dove sono salvati i file delle metriche
evaluation_folder = '../../data/evaluation'
experiments_neigh = [1, 2, 4, 6, 8, 10]
num_experiments_data = 10  # Numero di esperimenti

# Inizializza le liste per conservare le metriche medie per ogni numero di vicini
maes = {k: [] for k in experiments_neigh}
maes_users = {k: [] for k in experiments_neigh}
rmses = {k: [] for k in experiments_neigh}
rmses_users = {k: [] for k in experiments_neigh}
coverage_pos = {k: [] for k in experiments_neigh}
coverage_neg = {k: [] for k in experiments_neigh}
coverage_tot = {k: [] for k in experiments_neigh}

# Carica e aggrega le metriche da ogni file
for experiment in range(num_experiments_data):
    file_path = os.path.join(evaluation_folder, f'fold_{experiment}_metrics.pkl')
    with open(file_path, 'rb') as file:
        metrics = pickle.load(file)
        for k in experiments_neigh:
            maes[k].append(np.mean(metrics[k]['MAE']))
            maes_users[k].append(np.mean(metrics[k]['MAE_users']))
            rmses[k].append(np.mean(metrics[k]['RMSE']))
            rmses_users[k].append(np.mean(metrics[k]['RMSE_users']))
            coverage_pos[k].append(np.mean(metrics[k]['Coverage_Pos']))
            coverage_neg[k].append(np.mean(metrics[k]['Coverage_Neg']))
            coverage_tot[k].append(np.mean(metrics[k]['Coverage_Tot']))

# Neighborhood sizes
range_neigh = [1, 2, 4, 6, 8, 10]
rmses_means = [np.mean(rmses[k]) for k in experiments_neigh]
rmses_users_means = [np.mean(rmses_users[k]) for k in experiments_neigh]



# Set the positions for the bars on the x-axis
positions = range(len(range_neigh))

x_values = np.array(experiments_neigh)  # Usare la lista 'experiments_neigh' per l'asse x

# Creare il grafico
plt.figure(figsize=(10, 6))

# Disegnare le linee per MAE e MAE degli utenti
plt.plot(x_values, rmses_users_means,  label='RMSE', marker='o')

# Aggiungere legenda e etichette
plt.xlabel('Number of Neighbors', fontsize=14, fontweight='bold')
plt.ylabel('Root Mean Squared Error', fontsize=14, fontweight='bold')
plt.title('RMSE for Varying Number of Neighbors', fontsize=16, fontweight='bold')
plt.legend()

# Mostrare il grafico
plt.show()
