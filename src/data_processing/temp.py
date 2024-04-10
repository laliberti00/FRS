import pandas as pd
import numpy as np

test_data = pd.read_pickle('../../data/test_data/test0.pkl')
pred_data = pd.read_pickle('../../data/predictions/Predictions-0-1.pkl')


print(test_data.head())
print(pred_data.head())


print(pred_data['predicted_rating'])
print(test_data['rating'])







