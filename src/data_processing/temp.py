import src.data_processing.filter_divide_dataset
from src.data_processing.filter_divide_dataset import main as filter_divide_dataset
from src.fuzzy_user_signature.user_signature import main as user_signature
from src.fuzzy_user_signature.kindredness_calculation import main as kindredness_calculation
from src.predictions.predictions_maker import main as predictions_maker
from src.evaluation.predictions_evaluation import main as predictions_evaluation
import os
import pandas as pd
import numpy as np


def main():
    minimi = []
    massimi = []
    mediane = []
    medie = []
    primi_quartili = []
    terzi_quartili = []

    for n in range(10):
        cartella = f"../../data/user_signatures/fold{n}"

        for file in os.listdir(cartella):
            if file.endswith(".pkl"):
                percorso_file = os.path.join(cartella, file)
                dati = pd.read_pickle(percorso_file)
                np_dati = dati.values
                np_dati_filtrati = np_dati[np_dati != 0]
                minimi.append(np.min(np_dati_filtrati))
                massimi.append(np.max(np_dati_filtrati))
                mediane.append(np.median(np_dati_filtrati))
                medie.append(np.mean(np_dati_filtrati))
                primi_quartili.append(np.percentile(np_dati_filtrati, 0.25))
                terzi_quartili.append(np.percentile(np_dati_filtrati, 0.75))

        with open(f"statistiche{n}.txt", "w") as file:
            file.write(f"Media Minimo\t{np.mean(minimi).round(4)}\tDeviazione standard Minimo\t{np.std(minimi).round(4)}\n")
            file.write(f"Media Massimo\t{np.mean(massimi).round(4)}\tDeviazione standard Massimo\t{np.std(massimi).round(4)}\n")
            file.write(f"Media Mediana\t{np.mean(mediane).round(4)}\tDeviazione standard Mediana\t{np.std(mediane).round(4)}\n")
            file.write(f"Media Media\t{np.mean(medie).round(4)}\tDeviazione standard Media\t{np.std(medie).round(4)}\n")
            file.write(
                f"Media Primo Quartile\t{np.mean(primi_quartili).round(4)}\tDeviazione standard Primo Quartile\t{np.std(primi_quartili).round(4)}\n")
            file.write(
                f"Media Terzo Quartile\t{np.mean(terzi_quartili).round(4)}\tDeviazione standard Terzo Quartile\t{np.std(terzi_quartili).round(4)}\n")


if __name__ == "__main__":
    main()







