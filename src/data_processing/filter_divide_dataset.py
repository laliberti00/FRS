import os

import pandas as pd


def load_and_prepare_data(ratings_path, items_path, users_path):
    # Load ratings data
    ratings = pd.read_csv(ratings_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # Load items data and drop unnecessary columns
    items = pd.read_csv(items_path, sep='|', encoding='latin-1', header=None,
                        names=['movie_id', 'movie_title', 'release_date',
                               'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                               "Children's",
                               'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                               'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    items = items.drop(['release_date', 'video_release_date', 'IMDb_URL'], axis=1)

    # Load users data
    users = pd.read_csv(users_path, sep='|', encoding='latin-1', header=None,
                        names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

    return ratings, items, users


def filter_top_n_users(data, n=497):
    # Filter top-n users based on the number of ratings
    top_n_users = data.groupby("user_id").size().nlargest(n).index
    return data[data['user_id'].isin(top_n_users)], top_n_users


def filter_frequently_rated_movies(data, items, threshold=20):
    # Keep only movies rated more than the threshold
    movie_counts = data['movie_id'].value_counts()
    movies_to_keep = movie_counts[movie_counts >= threshold].index
    return data[data['movie_id'].isin(movies_to_keep)], items[items['movie_id'].isin(movies_to_keep)]


import numpy as np

def prepare_cross_validation_sets(data_users, users, folds=10, seed_base=42):
    np.random.seed(seed_base)  # Imposta un seed base per la riproducibilità
    results = []
    user_ids_shuffled = users['user_id'].values
    for i in range(folds):
        np.random.shuffle(user_ids_shuffled)  # Mischia gli ID degli utenti per ogni fold
        test_set = pd.DataFrame()
        training_set = pd.DataFrame()

        for user_id in user_ids_shuffled:
            user_data = data_users[data_users['user_id'] == user_id]
            split_point = int(len(user_data) * 0.1)
            if i < folds - 1:
                user_test = user_data.iloc[i * split_point:(i + 1) * split_point]
            else:
                # L'ultimo fold potrebbe essere più grande a causa dell'arrotondamento
                user_test = user_data.iloc[i * split_point:]
            user_train = pd.concat([user_data, user_test]).drop_duplicates(keep=False)

            test_set = pd.concat([test_set, user_test])
            training_set = pd.concat([training_set, user_train])

        results.append((training_set.reset_index(drop=True), test_set.reset_index(drop=True)))
    return results



def save_filtered_users(users, top_n_users_ids):
    # Save the filtered users to a pickle file
    if not os.path.exists('../../data/user_data/'):
        os.makedirs('../../data/user_data/')
    filtered_users = users[users['user_id'].isin(top_n_users_ids)]
    filtered_users.to_pickle('../../data/user_data/users.pkl')


def main():
    paths = {
        'ratings': '../../ml-100k/u.data',
        'items': '../../ml-100k/u.item',
        'users': '../../ml-100k/u.user'
    }

    ratings, items, users = load_and_prepare_data(paths['ratings'], paths['items'], paths['users'])
    ratings_items_merged = pd.merge(ratings, items, on='movie_id')
    filtered_ratings, top_n_users_ids = filter_top_n_users(ratings_items_merged)
    filtered_data, filtered_items = filter_frequently_rated_movies(filtered_ratings, items)

    if not os.path.exists('../../data/item_data/'):
        os.makedirs('../../data/item_data/')
    # Save filtered items
    filtered_items.to_pickle('../../data/item_data/items.pkl')

    # Save filtered users
    save_filtered_users(users, top_n_users_ids)

    if not os.path.exists('../../data/training_data/'):
        os.makedirs('../../data/training_data/')
    if not os.path.exists('../../data/test_data/'):
        os.makedirs('../../data/test_data/')
    # Prepare and save cross-validation sets
    cross_validation_sets = prepare_cross_validation_sets(filtered_data, users)
    for i, (train_set, test_set) in enumerate(cross_validation_sets):
        train_set.to_pickle(f'../../data/training_data/training{i}.pkl')
        test_set.to_pickle(f'../../data/test_data/test{i}.pkl')


