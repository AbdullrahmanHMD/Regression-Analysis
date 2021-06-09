from operator import index
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import pandas as pd
import os
import math

# Data retrieval from CSV using Pandas:
file_name = "Real_estate.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
real_estate_data = pd.read_csv(path).to_numpy()

# Splitting the data into features and labels
X = np.array(real_estate_data[:,1: -1])
Y = np.array(real_estate_data[:,-1])
N = len(X)

# getting rid of nan values if there any.
from RegressionUtils import replace_nan_values, split_data
X = replace_nan_values(X)

features = pd.read_csv(path).columns.to_numpy()


# Feartue selection: we will select the most affecting 3 features using Chi2 feature selection.
from RegressionUtils import select_n_features

n = 5
X, feature_indecies = select_n_features(real_estate_data, n)

# Retrieving the remaining features.
features = features[feature_indecies]

# Dimensionality reduction with PCA:
from RegressionUtils import dimensinality_reduction
# The number of dimensions after performing dimensionality
# reduction
n = 1
X = dimensinality_reduction(X, n)
X = np.reshape(X, (N, ))

from RegressionUtils import split_data
X_train, X_test, y_train, y_test = split_data(X, Y)


from PolynomialRegression import PolynomialRegression
from PolynomialRegression import predict
from RegressionUtils import RMSE
from RegressionUtils import mean_squared_errors



rmse_values = []
i = 1
K = 1  # degree of polynomial
K_values = []
# optimization by trial
while(i < 40):

    W = PolynomialRegression(X_train, y_train, K)
    y_pred = predict(X_test, W, K)
    rmse = RMSE(y_test, y_pred)
    rmse_values.append(rmse)
    K_values.append(K)
    K += 1
    i += 1

# obtaining best_K by getting the min RMSE from rmse_values
best_K = np.argmin(rmse_values) + 1 # we add one because we strated K from 1 and the array is 0-based indexing

# Training by
W = PolynomialRegression(X_train, y_train, best_K)
y_pred = predict(X_test, W, best_K)

## RMSE and MSE
RMSE = RMSE(y_test, y_pred)
MSE = mean_squared_errors(y_test, y_pred)
print("\nThe RMSE for Polynomial Regression with the " + file_name + " Dataset is ", RMSE, " with K = ", best_K)
print("The MSE for Polynomial Regression with the " + file_name + " Dataset is ", MSE, " with K = ", best_K, "\n")
print("Selected features for " + file_name + " Dataset:\n")

index = 1
for feature in features:
    print(str(index) + ") " + feature)
    index += 1

print()

#___________________________________________________________________________________________________________________________________________________________

X_t = linspace(min(X_train), max(X_train), len(X_train))

#Plotting
plt.plot(X, Y, "b.", markersize = 10)
plt.plot(X_t, predict(X_t, W, best_K), "r-")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#Plotting The RMSE
plt.plot(K_values, rmse_values, "b-")
plt.xlabel("K-degree")
plt.ylabel("RMSE")
plt.show()

#_________________________________________________________________________________________________________________________________________________________
# 
# For Second Dataset

# Data retrieval from CSV using Pandas:
file_name = "insurance.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
real_estate_data = pd.read_csv(path).to_numpy()

# Splitting the data into features and labels
X = np.array(real_estate_data[:,1: -1])
Y = np.array(real_estate_data[:,-1])
N = len(X)

# getting rid of nan values if there any.
from RegressionUtils import replace_nan_values, split_data
X = replace_nan_values(X)

features = pd.read_csv(path).columns.to_numpy()


# Feartue selection: we will select the most affecting 3 features using Chi2 feature selection.
from RegressionUtils import select_n_features

n = 3
X, feature_indecies = select_n_features(real_estate_data, n)

# Retrieving the remaining features.
features = features[feature_indecies]


# Dimensionality reduction with PCA:
from RegressionUtils import dimensinality_reduction
# The number of dimensions after performing dimensionality
# reduction
n = 1
X = dimensinality_reduction(X, n)
X = np.reshape(X, (N, ))

from RegressionUtils import split_data
X_train, X_test, y_train, y_test = split_data(X, Y)




from PolynomialRegression import PolynomialRegression
from PolynomialRegression import predict
from RegressionUtils import RMSE
from RegressionUtils import mean_squared_errors



rmse_values = []
i = 1
K = 1  # degree of polynomial
K_values = []
# optimization by trial
while(i < 40):

    W = PolynomialRegression(X_train, y_train, K)
    y_pred = predict(X_test, W, K)
    rmse = RMSE(y_test, y_pred)
    rmse_values.append(rmse)
    K_values.append(K)
    K += 1
    i += 1

# obtaining best_K by getting the min RMSE from rmse_values
best_K = np.argmin(rmse_values) + 1 # we add one because we strated K from 1 and the array is 0-based indexing

# Training by
W = PolynomialRegression(X_train, y_train, best_K)
y_pred = predict(X_test, W, best_K)

## RMSE and MSE
RMSE = RMSE(y_test, y_pred)
MSE = mean_squared_errors(y_test, y_pred)
print("The RMSE for Polynomial Regression with the " + file_name + " Dataset is ", RMSE, " with K = ", best_K)
print("The MSE for Polynomial Regression with the " + file_name + " Dataset is ", MSE, " with K = ", best_K, "\n")
print("Selected features for " + file_name + " Dataset:")

index = 1
for feature in features:
    print(str(index) + ") " + feature)
    index += 1

print()

#___________________________________________________________________________________________________________________________________________________________


X_t = linspace(min(X_train), max(X_train), len(X_train))

#Plotting
plt.plot(X, Y, "b.", markersize = 10)
plt.plot(X_t, predict(X_t, W, best_K), "r-")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#Plotting The RMSE
plt.plot(K_values, rmse_values, "b-")
plt.xlabel("K-degree")
plt.ylabel("RMSE")
plt.show()


#_________________________________________________________________________________________________________________________________________________________
# 
# For Third Dataset

# Data retrieval from CSV using Pandas:
file_name = "CarPrice_Assignment.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
real_estate_data = pd.read_csv(path).to_numpy()

# Splitting the data into features and labels
X = np.array(real_estate_data[:,1: -1])
Y = np.array(real_estate_data[:,-1])
N = len(X)

# getting rid of nan values if there any.
from RegressionUtils import replace_nan_values, split_data
X = replace_nan_values(X)

features = pd.read_csv(path).columns.to_numpy()


# Feartue selection: we will select the most affecting 3 features using Chi2 feature selection.
from RegressionUtils import select_n_features

n = 5
X, feature_indecies = select_n_features(real_estate_data, n)

# Retrieving the remaining features.
features = features[feature_indecies]


# Dimensionality reduction with PCA:
from RegressionUtils import dimensinality_reduction
# The number of dimensions after performing dimensionality
# reduction
n = 1
X = dimensinality_reduction(X, n)
X = np.reshape(X, (N, ))

from RegressionUtils import split_data
X_train, X_test, y_train, y_test = split_data(X, Y)




from PolynomialRegression import PolynomialRegression
from PolynomialRegression import predict
from RegressionUtils import RMSE
from RegressionUtils import mean_squared_errors



rmse_values = []
i = 1
K = 1  # degree of polynomial
K_values = []
# optimization by trial
while(i < 40):

    W = PolynomialRegression(X_train, y_train, K)
    y_pred = predict(X_test, W, K)
    rmse = RMSE(y_test, y_pred)
    rmse_values.append(rmse)
    K_values.append(K)
    K += 1
    i += 1

# obtaining best_K by getting the min RMSE from rmse_values
best_K = np.argmin(rmse_values) + 1 # we add one because we strated K from 1 and the array is 0-based indexing

# Training by
W = PolynomialRegression(X_train, y_train, best_K)
y_pred = predict(X_test, W, best_K)

## RMSE and MSE
RMSE = RMSE(y_test, y_pred)
MSE = mean_squared_errors(y_test, y_pred)
print("The RMSE for Polynomial Regression with the " + file_name + " Dataset is ", RMSE, " with K = ", best_K)
print("The MSE for Polynomial Regression with the " + file_name + " Dataset is ", MSE, " with K = ", best_K, "\n")
print("Selected features for " + file_name + " Dataset:")

index = 1
for feature in features:
    print(str(index) + ") " + feature)
    index += 1

print()

#___________________________________________________________________________________________________________________________________________________________

X_t = linspace(min(X_train), max(X_train), len(X_train))

#Plotting
plt.plot(X, Y, "b.", markersize = 10)
plt.plot(X_t, predict(X_t, W, best_K), "r-")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#Plotting The RMSE
plt.plot(K_values, rmse_values, "b-")
plt.xlabel("K-degree")
plt.ylabel("RMSE")
plt.show()
