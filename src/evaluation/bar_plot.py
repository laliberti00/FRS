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
            rmses[k].append(np.mean(metrics[k]['RMSE']))
            coverage_pos[k].append(np.mean(metrics[k]['Coverage_Pos']))
            coverage_neg[k].append(np.mean(metrics[k]['Coverage_Neg']))
            coverage_tot[k].append(np.mean(metrics[k]['Coverage_Tot']))




# Neighborhood sizes
range_neigh = [1, 2, 4, 6, 8, 10]
maes_means = [np.mean(maes[k]) for k in experiments_neigh]
maes_stds = [np.std(maes[k]) for k in experiments_neigh]

# Set the positions for the bars on the x-axis
positions = range(len(range_neigh))

plt.figure(figsize=(10, 6))

# Narrowing the width of the bars and reducing space between them for better visibility
width = 0.6
# Adjusting the bar chart to decrease the distance between bars on the x-axis and increase font sizes

# Setting positions closer together
positions = [i for i in range(len(range_neigh))]

# Creating the bar chart with closer bar positions
bars = plt.bar(positions, maes_means, width, yerr=maes_stds, capsize=6, color='skyblue')

# Adding deviation values as labels above the bars, with increased space for clarity
for bar, deviation in zip(bars, maes_stds):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + deviation + 0.005, f'{round(deviation, 2)}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Enhancing font sizes for better readability
plt.xticks(positions, range_neigh, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Number of Neighbors', fontsize=16, fontweight='bold')
plt.ylabel('Mean Absolute Error (MAE)', fontsize=16, fontweight='bold')
plt.title('Average MAE with Standard Deviation for Varying Neighbor', fontsize=18, fontweight='bold')

# Adjusting layout for better fit
plt.tight_layout()

# Show the plot
plt.show()
