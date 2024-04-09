import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import tqdm

# Funzione per caricare i risultati delle metriche da file
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
            coverage_pos[k].append(np.mean(metrics[k]['Coverage_Pos']))
            coverage_neg[k].append(np.mean(metrics[k]['Coverage_Neg']))
            coverage_tot[k].append(np.mean(metrics[k]['Coverage_Tot']))

# Neighborhood sizes
range_neigh = [1, 2, 4, 6, 8, 10]
maes_means = [np.mean(maes[k]) for k in experiments_neigh]
maes_users_means = [np.mean(maes_users[k]) for k in experiments_neigh]

cov_pos = [np.mean(coverage_pos[k])*100 for k in experiments_neigh]
cov_neg = [np.mean(coverage_neg[k])*100 for k in experiments_neigh]
cov_tot = [np.mean(coverage_tot[k])*100 for k in experiments_neigh]



# Set the positions for the bars on the x-axis
positions = range(len(range_neigh))

x_values = np.array(experiments_neigh)  # Usare la lista 'experiments_neigh' per l'asse x
# Calcolare i valori minimo e massimo tra tutti i dati
min_val = min(min(cov_pos), min(cov_neg), min(cov_tot))
max_val = max(max(cov_pos), max(cov_neg), max(cov_tot))

# Espandi leggermente i limiti per garantire che i dati siano ben visibili
margin = (max_val - min_val) * 0.05  # Aggiungi un margine del 5%
y_limits = (min_val - margin, max_val + margin)
# Ora crea i grafici
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 2)
plt.plot(range_neigh, cov_pos, 'k-D')
plt.ylim(y_limits)  # Applica gli stessi limiti dell'asse y
plt.title("Coverage POSITIVE", fontsize=16, fontweight='bold')
plt.xlabel("Number of neighbors", fontsize=14)
plt.ylabel("Coverage Rate", fontsize=14)

plt.subplot(1, 3, 3)
plt.plot(range_neigh, cov_neg, 'k-D')
plt.ylim(y_limits)  # Applica gli stessi limiti dell'asse y
plt.title("Coverage NEGATIVE", fontsize=16, fontweight='bold')
plt.xlabel("Number of neighbors", fontsize=14)
plt.ylabel("Coverage Rate", fontsize=14)

plt.subplot(1, 3, 1)
plt.plot(range_neigh, cov_tot, 'k-D', label="FUF Solution")
plt.ylim(y_limits)  # Applica gli stessi limiti dell'asse y
plt.title("Coverage TOTAL", fontsize=16, fontweight='bold')
plt.xlabel("Number of neighbors", fontsize=14)
plt.ylabel("Coverage Rate", fontsize=14)

plt.tight_layout()
plt.show()

Y_prob_COV = [83.5, 91.7, 93.6, 92.6, 92.6, 91.8]


plt.plot(range_neigh, cov_pos, 'k-D', label = "FUF Solution")
plt.plot(range_neigh, Y_prob_COV, 'r-o', label = "PF Solution")
plt.title("Coverage POSITIVE", fontsize=16, fontweight='bold')
plt.xlabel("Number of neighbors", fontsize=16, fontweight='bold')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(loc='best',
shadow=True, fontsize='small')

plt.show()
