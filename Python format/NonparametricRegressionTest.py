import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import pandas as pd
import os

from pandas.core.dtypes.missing import isnull

# Real_estate Data retrieval:
file_name = "Real_estate.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
real_estate_data = pd.read_csv(path).to_numpy()

X = np.array(real_estate_data[:,1: -1])
Y_truth = np.array(real_estate_data[:,-1]).reshape(-1 , 1)

# -----------------------------------------------------
# Real_estate Data Manipulation:

# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 3
X = select_n_features(real_estate_data, n)

# Dimensionality reduction:
from RegressionUtils import dimensinality_reduction
# The number of dimensions after performing dimensionality
# reduction
n = 1
X = dimensinality_reduction(X, n)

# Constructing train and test sets:
from sklearn.model_selection import train_test_split

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_truth, test_size = test_size, random_state=1)

#
# -----------------------------------------------------
# Regression:

from NonparametricRegression import kernel_smoother
x_t = linspace(min(X_train), max(X_train), len(X_train))
h = 150
Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

#
# -----------------------------------------------------
# Plotting Real_estate:

plt.plot(X_train, Y_train, "b.", markersize = 7)
plt.plot(X_test, Y_test, "g.", markersize = 5)

plt.plot(x_t, Y_pred, "r-")

plt.xlabel("x")
plt.ylabel("y")

# plt.show()

#
# -----------------------------------------------------

# insurance Data retrieval:
file_name = "insurance.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
insurance_data = pd.read_csv(path).to_numpy()

X = np.array(insurance_data[:-1])
Y_truth = np.array(insurance_data[:,-1]).reshape(-1 , 1)

# -----------------------------------------------------
# insurance Data Manipulation:

# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 5
X = select_n_features(insurance_data, n)

# Dimensionality reduction:
from RegressionUtils import dimensinality_reduction
# The number of dimensions after performing dimensionality
# reduction
n = 1
X = dimensinality_reduction(X, n)

# Constructing train and test sets:
from sklearn.model_selection import train_test_split

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_truth, test_size = test_size, random_state=1)

#
# -----------------------------------------------------
# Regression:

from NonparametricRegression import kernel_smoother
x_t = linspace(min(X_train), max(X_train), len(X_train))
h = 1
Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

#
# -----------------------------------------------------
# Plotting insurance:

plt.plot(X_train, Y_train, "b.", markersize = 7)
plt.plot(X_test, Y_test, "g.", markersize = 5)

plt.plot(x_t, Y_pred, "r-")

plt.xlabel("x")
plt.ylabel("y")

# plt.show()

#--------------------------------------------------------------------------------

# Life_Expectancy Data retrieval:
file_name = "Life_Expectancy_Data_New.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
life_expectancy_data = pd.read_csv(path).to_numpy()

X = np.array(life_expectancy_data[:,1: -1])
Y_truth = np.array(life_expectancy_data[:,-1]).reshape(-1 , 1)
print(Y_truth)
# Replacing NaN values with average column values.
from RegressionUtils import replace_nan_values
X = replace_nan_values(X)

# -----------------------------------------------------
# Life_Expectancy Data Manipulation:

# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 1
life_expectancy_data = replace_nan_values(life_expectancy_data)
X = select_n_features(life_expectancy_data, n)

# Dimensionality reduction:
from RegressionUtils import dimensinality_reduction
# The number of dimensions after performing dimensionality
# reduction
n = 1
X = dimensinality_reduction(X, n)

# Constructing train and test sets:
from sklearn.model_selection import train_test_split

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_truth, test_size = test_size, random_state=1)

#
# -----------------------------------------------------
# Regression:

from NonparametricRegression import kernel_smoother
x_t = linspace(min(X_train), max(X_train), len(X_train))
h = 50
Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

#
# -----------------------------------------------------
# Plotting Real_estate:

plt.plot(X_train, Y_train, "b.", markersize = 7)
plt.plot(X_test, Y_test, "g.", markersize = 5)

plt.plot(x_t, Y_pred, "r-")

plt.xlabel("x")
plt.ylabel("y")

plt.show()