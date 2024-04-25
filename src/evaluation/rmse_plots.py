import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

# Percorso alla cartella dove sono salvati i file delle metriche
evaluation_folder = '../../data/evaluation'
experiments_neigh = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
num_experiments_data = 10  # Numero di esperimenti

if not os.path.exists(f"../../data/plots"):
    os.makedirs(f"../../data/plots")

colors = ['b', 'g', 'r', 'c']  # Colori per le diverse linee di alpha
labels = [r'$\alpha=0$', r'$\alpha=0.25$', r'$\alpha=0.5$', r'$\alpha=0.7$']  # Etichette per la legenda
markers = ['o', 's', '^', 'd']  # Marker per i diversi alpha

# Creare il grafico
plt.figure(figsize=(10, 6))

for alpha in range(4):
    # Inizializza le liste per conservare le metriche medie per ogni numero di vicini
    rmses = {k: [] for k in experiments_neigh}
    rmses_users = {k: [] for k in experiments_neigh}

    # Carica e aggrega le metriche da ogni file
    for experiment in range(num_experiments_data):
        file_path = os.path.join(evaluation_folder, f'fold_{experiment}_{alpha}_metrics.pkl')
        with open(file_path, 'rb') as file:
            metrics = pickle.load(file)
            for k in experiments_neigh:
                rmses[k].append(np.mean(metrics[k]['RMSE']))
                rmses_users[k].append(np.mean(metrics[k]['RMSE_users']))

    # Neighborhood sizes
    range_neigh =  [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    rmses_means = [np.mean(rmses[k]) for k in experiments_neigh]
    rmses_users_means = [np.mean(rmses_users[k]) for k in experiments_neigh]

    # Disegnare le linee per RMSE e RMSE degli utenti
    plt.plot(experiments_neigh, rmses_means, label=labels[alpha] + ' (RMSE Data)', marker=markers[alpha],
             color=colors[alpha])
    plt.plot(experiments_neigh, rmses_users_means, label=labels[alpha] + ' (RMSE Users)', marker=markers[alpha],
             linestyle='--', color=colors[alpha])

# Aggiungere legenda e etichette
plt.xlabel('Number of Neighbors', fontsize=14, fontweight='bold')
plt.ylabel('Root Mean Squared Error', fontsize=14, fontweight='bold')
plt.title('RMSE for Varying Number of Neighbors', fontsize=16, fontweight='bold')
plt.legend()

plt.savefig(f'../../data/plots/rmse_combined.png')
