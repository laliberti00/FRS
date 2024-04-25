import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

# Percorso alla cartella dove sono salvati i file delle metriche
evaluation_folder = '../../data/evaluation'
experiments_neigh = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
num_experiments_data = 10  # Numero di esperimenti

# Inizializza le liste per conservare le metriche medie per ogni numero di vicini
cov_tot_alpha = {}

# Carica e aggrega le metriche da ogni file per ciascun valore di alpha
for alpha, a in enumerate([0, 0.25, 0.5, 0.7]):
    cov_tot_alpha[alpha] = []
    for experiment in range(num_experiments_data):
        file_path = os.path.join(evaluation_folder, f'fold_{experiment}_{alpha}_metrics.pkl')
        with open(file_path, 'rb') as file:
            metrics = pickle.load(file)
            cov_tot_alpha[alpha].append([np.mean(metrics[k]['Coverage_Tot']) * 100 for k in experiments_neigh])

# Calcolare la media per ciascun valore di experiments_neigh
cov_tot_mean = {alpha: np.mean(np.array(cov_tot_alpha[alpha]), axis=0) for alpha in range(4)}

# Creazione del grafico sovrapposto
plt.figure(figsize=(10, 6))

for alpha, a in enumerate([0, 0.25, 0.5, 0.7]):
    plt.plot(experiments_neigh, cov_tot_mean[alpha], label=f'Alpha={a}', marker='o')

# Aggiunta di legenda e etichette
plt.xlabel('Number of Neighbors', fontsize=14, fontweight='bold')
plt.ylabel('Coverage Rate (%)', fontsize=14, fontweight='bold')
plt.title('Coverage Total for Varying Number of Neighbors (Alpha=0,0.25,0.5,0.7)', fontsize=16, fontweight='bold')
plt.legend()

# Mostrare il grafico sovrapposto
plt.savefig(f'../../data/plots/coverage_total_combined.png')
