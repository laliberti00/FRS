import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

# Percorso alla cartella dove sono salvati i file delle metriche
evaluation_folder = '../../data/evaluation'
#experiments_neigh = [1, 2, 4, 6, 8, 10]
experiments_neigh = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
num_experiments_data = 10  # Numero di esperimenti

if not os.path.exists(f"../../data/plots"):
    os.makedirs(f"../../data/plots")

# Inizializza le liste per conservare le metriche medie per ogni numero di vicini
plt.figure(figsize=(10, 6))

colors = ['b', 'g', 'r', 'c']  # Colori per le diverse linee di alpha
labels = [r'$\alpha=0$', r'$\alpha=0.25$', r'$\alpha=0.5$', r'$\alpha=0.7$']  # Etichette per la legenda
markers = ['o', 's', '^', 'd']  # Marker per i diversi alpha

for alpha in range(4):
    maes = {k: [] for k in experiments_neigh}
    maes_users = {k: [] for k in experiments_neigh}

    # Carica e aggrega le metriche da ogni file
    for experiment in range(num_experiments_data):
        file_path = os.path.join(evaluation_folder, f'fold_{experiment}_{alpha}_metrics.pkl')
        with open(file_path, 'rb') as file:
            metrics = pickle.load(file)
            for k in experiments_neigh:
                maes[k].append(np.mean(metrics[k]['MAE']))
                maes_users[k].append(np.mean(metrics[k]['MAE_users']))

    # Calcola le medie per ciascun numero di vicini
    maes_means = [np.mean(maes[k]) for k in experiments_neigh]
    maes_users_means = [np.mean(maes_users[k]) for k in experiments_neigh]

    # Disegna le linee per MAE e MAE degli utenti
    plt.plot(experiments_neigh, maes_means, label=labels[alpha] + ' (MAE Data)', marker=markers[alpha], color=colors[alpha])
    plt.plot(experiments_neigh, maes_users_means, label=labels[alpha] + ' (MAE Users)', marker=markers[alpha], linestyle='--', color=colors[alpha])

# Aggiungi legenda e etichette
plt.xlabel('Number of Neighbors', fontsize=14, fontweight='bold')
plt.ylabel('Mean Absolute Error', fontsize=14, fontweight='bold')
plt.title('MAE Data and MAE Users for Varying Number of Neighbors', fontsize=16, fontweight='bold')
plt.legend()

# Salva il grafico
plt.savefig(f'../../data/plots/mae_combined.png')


Y_prob_MAE = [0.938, 0.815, 0.812, 0.824, 0.899, 0.910 ]
Y_prob_COV = [83.5, 91.7, 93.6, 92.6, 92.6, 91.8]


#plt.plot(range_neigh, maes_users_means, 'k-D', label = "FUF Solution")
#plt.plot(range_neigh, Y_prob_MAE, 'r-o', label = "PF Solution")
#plt.title("MAE_users", fontsize=16, fontweight='bold')
#plt.xlabel("Number of neighbors", fontsize=16, fontweight='bold')
#plt.yticks(fontsize=14)
#plt.xticks(fontsize=14)

#plt.legend(loc='best',
#shadow=True, fontsize='small')

#plt.show()



