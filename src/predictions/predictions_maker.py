import pandas as pd
from tqdm import tqdm
import os

# Definisce una funzione per caricare i dati necessari per un determinato esperimento.
def load_data(experiment):
    # Carica i dati di allenamento da un file pickle.
    training_data = pd.read_pickle(f'../../data/training_data/training{experiment}.pkl')
    # Carica i dati di test da un file pickle.
    test_data = pd.read_pickle(f'../../data/test_data/test{experiment}.pkl')
    # Carica i dati sui vicini da un file pickle.
    neighbors_data = pd.read_pickle(f'../../data/neighbors_data/F-neighbors-{experiment}.pkl')
    # Restituisce i tre set di dati caricati.
    return training_data, test_data, neighbors_data

# Definisce una funzione per calcolare una previsione di valutazione per un dato film e utente.
def make_prediction(user_id, movie_id, training_data, neighbors):
    # Filtra i dati di allenamento per ottenere solo le valutazioni dell'utente specificato.
    user_ratings = training_data[training_data['user_id'] == user_id]
    # Calcola la media delle valutazioni dell'utente.
    avg_user_rating = user_ratings['rating'].mean()

    # Inizializza una somma ponderata e una somma delle similarità.
    weighted_sum = 0
    sum_similarities = 0
    # Itera sui vicini e le loro similarità.
    for neighbor_id, similarity in neighbors:
        # Filtra le valutazioni del vicino specificato.
        neighbor_ratings = training_data[training_data['user_id'] == neighbor_id]
        neighbor_rating = neighbor_ratings[neighbor_ratings['movie_id'] == movie_id]['rating']
        # Se il vicino ha valutato il film, calcola il contributo alla somma ponderata.
        if not neighbor_rating.empty:
            avg_neighbor_rating = neighbor_ratings['rating'].mean()
            weighted_sum += (neighbor_rating.iloc[0] - avg_neighbor_rating) * similarity
            sum_similarities += abs(similarity)

    # Calcola la previsione basandosi sulla media dell'utente e sulla somma ponderata.
    prediction = avg_user_rating
    if sum_similarities > 0:
        prediction += weighted_sum / sum_similarities

    # Restituisce la previsione arrotondata.
    return round(prediction)

# Definisce una funzione per calcolare tutte le previsioni per un determinato esperimento.
def calc_all_predictions(experiment, num_neighbors):
    # Carica i dati necessari.
    training_data, test_data, neighbors_data = load_data(experiment)

    # Lista per conservare tutte le previsioni.
    predictions = []
    # Itera su ogni riga dei dati di test.
    for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
        # Estrae l'ID dell'utente e del film dalla riga corrente dei dati di test.
        user_id = row['user_id']
        movie_id = row['movie_id']
        # Estrae i primi N vicini per l'utente corrente.
        user_neighbors = neighbors_data[neighbors_data['user'] == user_id]['neighbors'].iloc[0][:num_neighbors]
        # Calcola la previsione per l'utente e il film correnti.
        predicted_rating = make_prediction(user_id, movie_id, training_data, user_neighbors)

        # Aggiunge la previsione alla lista delle previsioni.
        predictions.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'predicted_rating': predicted_rating
        })

    # Verifica se la directory di output esiste; in caso contrario, la crea.
    output_dir = f'../../data/predictions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Salva le previsioni in un DataFrame e poi in un file pickle.
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_pickle(f'{output_dir}/Predictions-{experiment}-{num_neighbors}.pkl')

    # Stampa un messaggio di conferma.
    print(f"Predictions for experiment {experiment} with {num_neighbors} neighbors have been saved.")

# Punto di ingresso principale dello script.
if __name__ == "__main__":
    # Definisce il range di vicini da considerare.
    experiments_neigh = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    # Itera su ogni esperimento (assumendo 10 esperimenti in totale).
    for experiment in range(10):
        print(f"########### Starting experiment {experiment} ###########")
        for num_neighbors in experiments_neigh:
            print(f"Calculating predictions with {num_neighbors} neighbors...")
            # Calcola le previsioni per l'esperimento corrente e il numero di vicini specificato.
            calc_all_predictions(experiment, num_neighbors)
