import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import tqdm

# Funzione per caricare i risultati delle metriche da file
def load_metrics(experiment, num_neighbors):
    evaluation_folder = '../../data/evaluation'
    filename = f'fold_{experiment}_metrics.pkl'
    file_path = os.path.join(evaluation_folder, filename)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            fold_metrics = pickle.load(f)
            return fold_metrics[num_neighbors]
    else:
        print(f"No data found for experiment {experiment}, neighbors {num_neighbors}.")
        return None

# Definisci il range di vicini e i numeri degli esperimenti
neighbors_range = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
experiments_range = list(range(10))

# Dizionari per memorizzare i dati aggregati
aggregated_metrics = {'MAE': [], 'RMSE': [], 'Coverage_Pos': []}

for num_neighbors in neighbors_range:
    mae_values = []
    rmse_values = []
    coverage_pos_values = []

    for experiment in experiments_range:
        metrics = load_metrics(experiment, num_neighbors)
        if metrics:
            mae_values.extend(metrics['MAE'])
            rmse_values.extend(metrics['RMSE'])
            coverage_pos_values.extend(metrics['Coverage_Pos'])

    aggregated_metrics['MAE'].append(np.mean(mae_values))
    aggregated_metrics['RMSE'].append(np.mean(rmse_values))
    aggregated_metrics['Coverage_Pos'].append(np.mean(coverage_pos_values))

# Ora crea i grafici
plt.figure(figsize=(14, 5))

# Grafico per MAE
plt.subplot(1, 3, 1)
plt.plot(neighbors_range, aggregated_metrics['MAE'], 'k-D')
plt.title("MAE", fontsize=16, fontweight='bold')
plt.xlabel("Number of neighbors", fontsize=14)
plt.ylabel("Mean Absolute Error", fontsize=14)

# Grafico per RMSE
plt.subplot(1, 3, 2)
plt.plot(neighbors_range, aggregated_metrics['RMSE'], 'r-D')
plt.title("RMSE", fontsize=16, fontweight='bold')
plt.xlabel("Number of neighbors", fontsize=14)
plt.ylabel("Root Mean Square Error", fontsize=14)

# Grafico per Coverage Positive
plt.subplot(1, 3, 3)
plt.plot(neighbors_range, aggregated_metrics['Coverage_Pos'], 'b-D', label="FUF Solution")
plt.title("Coverage Positive", fontsize=16, fontweight='bold')
plt.xlabel("Number of neighbors", fontsize=14)
plt.ylabel("Coverage Rate", fontsize=14)

plt.tight_layout()
plt.show()
