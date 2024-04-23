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
            predicted_rating = pred[2]
            if predicted_rating == 0:
                continue
            error = abs(predicted_rating - true_rating)
            errors.append(error)

    if errors:  # Verifica che la lista degli errori non sia vuota
        mae = np.sum(errors) / len(errors)
    else:
        mae = None  # o un altro valore di errore appropriato
    return round(mae, 3) if mae is not None else None


def calc_rmse(predictions, user_test):
    # Filtra predictions per considerare solo quelle con un predicted_rating > 0
    predictions_filtered = predictions[predictions['predicted_rating'] > 0]

    # Assicurati che user_test sia indicizzato per movie_id per un accesso efficiente
    user_test = user_test.set_index('movie_id')

    # Unisci predictions_filtered con user_test per assicurare che solo le righe con valori predetti > 0 siano considerate
    merged = predictions_filtered.join(user_test, on='movie_id', how='inner', lsuffix='_pred', rsuffix='_actual')

    # Calcola RMSE solo se merged non è vuoto
    if not merged.empty:
        mse = ((merged['predicted_rating'] - merged['rating']) ** 2).mean()
        rmse = math.sqrt(mse)
        return round(rmse, 3)
    else:
        return None


def calc_coverage(user_test, predictions):
    correct_pred_pos = 0
    correct_pred_neg = 0
    total_test_len_pos = 0
    total_test_len_neg = 0
    correct_pred_tot = 0
    total_test_len = user_test.shape[0]

    for index, row in predictions.iterrows():

        if row['movie_id'] in user_test['movie_id'].values:
            true_rating = user_test[user_test['movie_id'] == row['movie_id']]['rating'].values[0]
            predicted_rating = row['predicted_rating']

            if predicted_rating > 0:
                correct_pred_tot += 1

            '''
            # Per valutazioni positive
            if true_rating >= 3:
                total_test_len_pos += 1
                if predicted_rating > 0:
                    correct_pred_pos += 1
                    correct_pred_tot += 1

            # Per valutazioni negative
            elif true_rating < 3:
                total_test_len_neg += 1
                if predicted_rating < 3:
                    correct_pred_neg += 1
                    correct_pred_tot += 1
        else:
            print('OUT')
            '''


    #coverage_pos = correct_pred_pos / total_test_len_pos if total_test_len_pos > 0 else 0
    #coverage_neg = correct_pred_neg / total_test_len_neg if total_test_len_neg > 0 else 0
    coverage_tot = correct_pred_tot / total_test_len if total_test_len > 0 else 0

    return 0, 0, coverage_tot


def main():
    experiments_neigh = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    #experiments_neigh = [1, 2, 4, 6, 8, 10]
    num_experiments_data = list(range(10))

    for alpha in range(4):
        print(f'######### ALPHA {alpha} #########')
        overall_metrics = {}

        for experiment in tqdm(num_experiments_data):
            fold_metrics = {
                num: {'MAE': [], 'MAE_users': [], 'RMSE': [], 'RMSE_users': [], 'Coverage_Pos': [], 'Coverage_Neg': [], 'Coverage_Tot': []}
                for num
                in experiments_neigh}

            evaluation_folder = '../../data/evaluation'

            if not os.path.exists(evaluation_folder):
                os.makedirs(evaluation_folder)

            for num_neighbors in experiments_neigh:
                predictions_file = f'../../data/predictions/Predictions-{experiment}-{num_neighbors}-{alpha}.pkl'
                test_file = f'../../data/test_data/test{experiment}.pkl'

                predictions = pd.read_pickle(predictions_file)
                test_set = pd.read_pickle(test_file)

                mae = calc_mae(predictions, test_set)
                rmse = calc_rmse(predictions, test_set)
                coverage_pos, coverage_neg, coverage_tot = calc_coverage(test_set, predictions)

                fold_metrics[num_neighbors]['MAE'].append(mae)
                fold_metrics[num_neighbors]['RMSE'].append(rmse)
                fold_metrics[num_neighbors]['Coverage_Pos'].append(coverage_pos)
                fold_metrics[num_neighbors]['Coverage_Neg'].append(coverage_neg)
                fold_metrics[num_neighbors]['Coverage_Tot'].append(coverage_tot)


                grouped_predictions = predictions.groupby('user_id')
                individual_mae_users = []
                individual_rmse_users = []

                for user_id, group in grouped_predictions:

                    user_test = test_set[test_set['user_id'] == user_id]
                    user_predictions = group[['user_id', 'movie_id', 'predicted_rating']].reset_index(drop=True)

                    mae_users = calc_mae(user_predictions, user_test)
                    if mae_users is not None:
                        individual_mae_users.append(mae_users)
                    rmse_users = calc_rmse(user_predictions, user_test)
                    if rmse_users is not None:
                        individual_rmse_users.append(rmse_users)

                fold_metrics[num_neighbors]['MAE_users'] = np.mean(individual_mae_users)
                fold_metrics[num_neighbors]['RMSE_users'] = np.mean(individual_rmse_users)

            results_filename = f'fold_{experiment}_{alpha}_metrics.pkl'
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
            num: {'MAE': [], 'MAE_users': [], 'RMSE': [],'RMSE_users': [], 'Coverage_Pos': [], 'Coverage_Neg': [], 'Coverage_Tot': []} for
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
