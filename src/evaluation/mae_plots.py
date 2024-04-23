import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

# Percorso alla cartella dove sono salvati i file delle metriche
evaluation_folder = '../../data/evaluation'
#experiments_neigh = [1, 2, 4, 6, 8, 10]
experiments_neigh = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
num_experiments_data = 10  # Numero di esperimenti

# Inizializza le liste per conservare le metriche medie per ogni numero di vicini
for alpha in range(4):
    maes = {k: [] for k in experiments_neigh}
    maes_users = {k: [] for k in experiments_neigh}
    rmses = {k: [] for k in experiments_neigh}
    coverage_pos = {k: [] for k in experiments_neigh}
    coverage_neg = {k: [] for k in experiments_neigh}
    coverage_tot = {k: [] for k in experiments_neigh}

    # Carica e aggrega le metriche da ogni file
    for experiment in range(num_experiments_data):
        file_path = os.path.join(evaluation_folder, f'fold_{experiment}_{alpha}_metrics.pkl')
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
    #range_neigh = [1, 2, 4, 6, 8, 10]
    range_neigh = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    maes_means = [np.mean(maes[k]) for k in experiments_neigh]
    maes_users_means = [np.mean(maes_users[k]) for k in experiments_neigh]


    # Set the positions for the bars on the x-axis
    positions = range(len(range_neigh))

    x_values = np.array(experiments_neigh)  # Usare la lista 'experiments_neigh' per l'asse x

    # Creare il grafico
    plt.figure(figsize=(10, 6))

    # Disegnare le linee per MAE e MAE degli utenti
    plt.plot(x_values, maes_means, label='MAE Data', marker='o')
    plt.plot(x_values, maes_users_means, label='MAE Users', marker='s')

    # Aggiungere legenda e etichette
    plt.xlabel('Number of Neighbors', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Absolute Error', fontsize=14, fontweight='bold')
    plt.title('MAE Data and MAE Users for Varying Number of Neighbors', fontsize=16, fontweight='bold')
    plt.legend()

    # Mostrare il grafico
    plt.show()


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



