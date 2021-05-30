from operator import index
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Data retrieval:
file_name = "Real_estate.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
real_estate_data = pd.read_csv(path).to_numpy()

X = np.array(real_estate_data[:,1: -1])
Y_truth = np.array(real_estate_data[:,-1])

# Feartue selection.
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 3
X = select_n_features(real_estate_data, n)
