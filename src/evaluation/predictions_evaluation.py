import pandas as pd
import numpy as np
import os
import math
from tqdm import tqdm
import pickle


def calc_mae(predictions, user_test):

    errors = []
    for pred in predictions.values:
        filtered_user_test = user_test.loc[user_test["movie_id"] == pred[1], 'rating']

        if not filtered_user_test.empty:
            true_rating = filtered_user_test.values[0]
            predicted_rating = pred[2]  # Assicurati che questo sia l'indice corretto per il rating predetto
            error = abs(predicted_rating - true_rating)
            errors.append(error)

    if errors:  # Verifica che la lista degli errori non sia vuota
        mae = np.sum(errors) / len(errors)
    else:
        mae = None  # o un altro valore di errore appropriato
    return round(mae, 3) if mae is not None else None


def calc_rmse(predictions, user_test):
    # Assicurati che user_test sia indicizzato per movie_id per un accesso efficiente
    user_test = user_test.set_index('movie_id')

    # Ottieni solo le righe di user_test che corrispondono ai movie_id in predictions
    matched_ratings = user_test.loc[predictions['movie_id']]['rating']

    # Calcola RMSE
    mse = ((predictions['predicted_rating'] - matched_ratings) ** 2).mean()
    rmse = math.sqrt(mse)

    return round(rmse, 3)


def calc_coverage(user_test, predictions):
    correct_pred_pos = 0
    correct_pred_neg = 0
    total_test_len_pos = 0
    total_test_len_neg = 0
    correct_pred_tot = 0
    total_test_len = user_test.shape[0]

    for index, row in predictions.iterrows():
        if row['movie_id'] in user_test['movie_id'].values:
            true_rating = user_test[user_test['movie_id'] == row['movie_id']]['predicted_rating'].values[0]
            predicted_rating = row['rating']

            # Per valutazioni positive
            if true_rating >= 3:
                total_test_len_pos += 1
                if predicted_rating >= 3:
                    correct_pred_pos += 1
                    correct_pred_tot += 1

            # Per valutazioni negative
            elif true_rating < 3:
                total_test_len_neg += 1
                if predicted_rating < 3:
                    correct_pred_neg += 1
                    correct_pred_tot += 1

    coverage_pos = correct_pred_pos / total_test_len_pos if total_test_len_pos > 0 else 0
    coverage_neg = correct_pred_neg / total_test_len_neg if total_test_len_neg > 0 else 0
    coverage_tot = correct_pred_tot / total_test_len if total_test_len > 0 else 0

    return coverage_pos, coverage_neg, coverage_tot


def main():
    experiments_neigh = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    num_experiments_data = list(range(10))

    overall_metrics = {}

    for experiment in tqdm(num_experiments_data):
        fold_metrics = {
            num: {'MAE': [], 'MAE_users': [], 'RMSE': [], 'Coverage_Pos': [], 'Coverage_Neg': [], 'Coverage_Tot': []}
            for num
            in experiments_neigh}

        evaluation_folder = '../../data/evaluation'

        if not os.path.exists(evaluation_folder):
            os.makedirs(evaluation_folder)

        for num_neighbors in experiments_neigh:
            predictions_file = f'../../data/predictions/Predictions-{experiment}-{num_neighbors}.pkl'
            test_file = f'../../data/test_data/test{experiment}.pkl'

            predictions = pd.read_pickle(predictions_file)
            test_set = pd.read_pickle(test_file)

            mae = calc_mae(predictions, test_set)
            rmse = calc_rmse(predictions, test_set)
            coverage_pos, coverage_neg, coverage_tot = calc_coverage(predictions, test_set)

            fold_metrics[num_neighbors]['MAE'].append(mae)
            fold_metrics[num_neighbors]['RMSE'].append(rmse)
            fold_metrics[num_neighbors]['Coverage_Pos'].append(coverage_pos)
            fold_metrics[num_neighbors]['Coverage_Neg'].append(coverage_neg)
            fold_metrics[num_neighbors]['Coverage_Tot'].append(coverage_tot)


            grouped_predictions = predictions.groupby('user_id')
            individual_mae_users = []

            for user_id, group in grouped_predictions:

                user_test = test_set[test_set['user_id'] == user_id]
                user_predictions = group[['user_id', 'movie_id', 'predicted_rating']].reset_index(drop=True)

                mae_users = calc_mae(user_predictions, user_test)
                individual_mae_users.append(mae_users)

            fold_metrics[num_neighbors]['MAE_users'] = np.mean(individual_mae_users)

        results_filename = f'fold_{experiment}_metrics.pkl'
        with open(os.path.join(evaluation_folder, results_filename), 'wb') as f:
            pickle.dump(fold_metrics, f)
        overall_metrics[experiment] = fold_metrics

    print(overall_metrics)

    # Analisi e stampa dei risultati per ogni fold
    for experiment, metrics in overall_metrics.items():
        print(f"\nResults for Fold {experiment}:")
        for num_neighbors, values in metrics.items():
            print(f"\nNumber of Neighbors: {num_neighbors}")
            for metric, results in values.items():
                mean_val = np.mean(results)
                std_val = np.std(results)
                print(f"{metric}: Mean = {mean_val:.3f}, Std = {std_val:.3f}")

    # Calcola e stampa i risultati aggregati su tutti i fold
    aggregated_results = {
        num: {'MAE': [], 'MAE_users': [], 'RMSE': [], 'Coverage_Pos': [], 'Coverage_Neg': [], 'Coverage_Tot': []} for
        num in experiments_neigh}
    for experiment, metrics in overall_metrics.items():
        for num_neighbors, values in metrics.items():
            for metric, results in values.items():
                if isinstance(results, list):  # Se results è una lista
                    aggregated_results[num_neighbors][metric].extend(results)
                else:  # Se results è un singolo valore
                    aggregated_results[num_neighbors][metric].append(results)

    print("\nAggregated Results Across All Folds:")
    for num_neighbors, values in aggregated_results.items():
        print(f"\nNumber of Neighbors: {num_neighbors}")
        for metric, results in values.items():
            mean_val = np.mean(results)
            std_val = np.std(results)
            print(f"{metric}: Mean = {mean_val:.3f}, Std = {std_val:.3f}")


if __name__ == "__main__":
    main()
