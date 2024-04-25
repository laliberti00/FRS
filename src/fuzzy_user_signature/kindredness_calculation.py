import os

import pandas as pd
from tqdm import tqdm
from heapq import nlargest
from operator import itemgetter
import numpy as np
import sys

def load_filter_data(num_train):
    data_users = pd.read_pickle(f'../../data/training_data/training{num_train}.pkl')
    users = pd.read_pickle('../../data/user_data/users.pkl')
    return data_users, users

def calc_similarity(user, neighbor, alpha=0):
    user_aligned, neighbor_aligned = user.align(neighbor, join='inner', axis=1)
    user_array = user_aligned.values
    neighbor_array = neighbor_aligned.values
    t_norm_array = np.minimum(user_array, neighbor_array)
    # obtain alpha cut
    if alpha > 0:
        t_norm_filtered = np.where(t_norm_array > alpha, t_norm_array, 0)
        user_array_filtered = np.where(user_array > alpha, user_array, 0)
    else:
        t_norm_filtered = t_norm_array
        user_array_filtered = user_array
    summ = np.sum(t_norm_filtered)
    sum_user = np.sum(user_array_filtered)
    return round(summ / sum_user, 2)

def main(trains):
    alpha_cuts = [0, 0.25, 0.5, 0.7]
    for num_train in range(trains):
        data_users, users = load_filter_data(num_train)
        print(f'## Experiment {num_train}')
        for k, a in enumerate(alpha_cuts):
            data_neigh = []
            print(f'# Alpha {a} {k}')
            if not os.path.exists(f"../../data/neighbors_data/alpha{k}"):
                os.makedirs(f"../../data/neighbors_data/alpha{k}")

            for i in tqdm(range(len(users))):  # Usa range(len(users)) per iterare sugli indici
                user_id = users.iloc[i]['user_id']
                user_interest = pd.read_pickle(f'../../data/user_signatures/fold{num_train}/f_user_sign{user_id}.pkl')

                similarities = []
                for j in range(len(users)):  # Stesso uso di range(len(users)) per iterare
                    if i != j:
                        new_user_id = users.iloc[j]['user_id']
                        neighbor = pd.read_pickle(f'../../data/user_signatures/fold{num_train}/f_user_sign{new_user_id}.pkl')
                        similarity = calc_similarity(user_interest, neighbor, a)
                        similarities.append((new_user_id, similarity))

                neighbors = nlargest(50, similarities, key=itemgetter(1))
                data_neigh.append({'user': user_id, 'neighbors': neighbors})

            usersData = pd.DataFrame(data_neigh)
            usersData.to_pickle(f"../../data/neighbors_data/alpha{k}/F-neighbors-{num_train}.pkl")

if __name__ == "__main__":
    main(10)
