import numpy as np
import pandas as pd

from titanic_visualizations import survival_stats
from IPython.display import display

in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

outcomes = full_data['Survived']
data = full_data.drop('Survived',axis=1)

def accuracy_score (truth,pred):
    if len(truth)==len(pred):
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean() * 100)
    else:
        return "Number of predictions does not match number of outcomes!"

