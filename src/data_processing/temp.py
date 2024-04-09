import pandas as pd

filtered_users = pd.read_pickle('../../data/user_data/users.pkl')

print(len(filtered_users))



import os

# Percorso della cartella da esaminare
directory_path = '/Users/lucaaliberti/PycharmProjects/FuzzyRecommenderSystem/data/user_signatures/fold0'

# Ottiene la lista di tutti i file e le cartelle nel percorso specificato
all_items = os.listdir(directory_path)

# Filtra la lista per includere solo i file (escludendo le cartelle)
files = [item for item in all_items if os.path.isfile(os.path.join(directory_path, item))]

# Stampa il numero di file presenti nella cartella
print(f"Il numero di file nella cartella '{directory_path}' è: {len(files)}")
