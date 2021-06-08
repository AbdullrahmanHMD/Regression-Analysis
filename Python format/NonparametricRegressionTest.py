import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import pandas as pd
import os

# Real_estate Data retrieval:
file_name = "Real_estate.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name

real_estate_data = pd.read_csv(path).to_numpy()
features = pd.read_csv(path).columns.to_numpy()

X = np.array(real_estate_data[:: -1])
Y_truth = np.array(real_estate_data[:,-1]).reshape(-1 , 1)

# Replacing NaN values with average column values.
from RegressionUtils import replace_nan_values

X = replace_nan_values(X)

# -----------------------------------------------------
# Real_estate Data Manipulation:

# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 3
X, feature_indecies = select_n_features(real_estate_data, n)

# Retrieving the remaining features.
features = features[feature_indecies]

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

X_train = np.array(X_train).reshape(1, -1)[0]
X_test = np.array(X_test).reshape(1, -1)[0]
Y_train = np.array(Y_train).reshape(1, -1)[0]
Y_test = np.array(Y_test).reshape(1, -1)[0]

#
# -------------------------------------------------------
# Regression:
from RegressionUtils import RMSE
from NonparametricRegression import kernel_smoother
x_t = linspace(min(X_train), max(X_train), len(X_train))
x_t_test = linspace(min(X_test), max(X_test), len(X_test))

epsilon = 0.001
rmse_values = [0, epsilon + 0.1]
i = 1
h = 100
h_increment = 1

while(abs(rmse_values[i] - rmse_values[i - 1]) > epsilon):

    Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

    Y_pred_test = kernel_smoother(X_test, x_t_test, Y_test, h)
    rmse = RMSE(Y_test, Y_pred_test)
    rmse_values.append(rmse)

    h += h_increment
    i += 1

Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

# Calculating RMSE:
Y_pred_test = kernel_smoother(X_test, x_t_test, Y_test, h)
rmse = RMSE(Y_test, Y_pred_test)

print("The RMSE for the kernel smoother with the Real_estate dataset is {} with h = {}".format(rmse, h))

# Calculating Squared Errors:
from RegressionUtils import mean_squared_errors

x_t_test = linspace(min(X_test), max(X_test), len(X_test))
Y_pred_test = kernel_smoother(X_test, x_t_test, Y_test, h)
mse = mean_squared_errors(Y_test, Y_pred_test)

print("The squared errors for the kernel smoother with the Real_estate dataset is {} \n".format(mse))

# Printing selected features:
print('Selected features for Real Estate data set:')
[print("{}) {}".format(i + 1, feature)) for i, feature in zip(range(len(features)),features)]
print("")

#
# -----------------------------------------------------
# Plotting Real_estate:

plt.plot(X_train, Y_train, "b.", markersize = 7)
plt.plot(X_test, Y_test, "g.", markersize = 5)

plt.plot(x_t, Y_pred, "r-")

plt.title("Real Estate")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Plotting h optimization.
x = linspace(0, len(rmse_values), len(rmse_values))

plt.plot(x, rmse_values, "r-")

plt.title("Error Function")
plt.xlabel("x")
plt.ylabel("y")

plt.show()
#
# -----------------------------------------------------

# insurance Data retrieval:
file_name = "insurance.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
insurance_data = pd.read_csv(path).to_numpy()

X = np.array(insurance_data[:-1])
Y_truth = np.array(insurance_data[:,-1]).reshape(-1 , 1)

X = replace_nan_values(X)

features = pd.read_csv(path).columns.to_numpy()

# -----------------------------------------------------
# insurance Data Manipulation:

# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 5 features.
n = 5
X, feature_indecies = select_n_features(insurance_data, n)

# Retrieving the remaining features.
features = features[feature_indecies]

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

X_train = np.array(X_train).reshape(1, -1)[0]
X_test = np.array(X_test).reshape(1, -1)[0]
Y_train = np.array(Y_train).reshape(1, -1)[0]
Y_test = np.array(Y_test).reshape(1, -1)[0]

#
# -----------------------------------------------------
# Regression:
from RegressionUtils import RMSE
from NonparametricRegression import kernel_smoother
x_t = linspace(min(X_train), max(X_train), len(X_train))
x_t_test = linspace(min(X_test), max(X_test), len(X_test))

epsilon = 0.001
rmse_values = [0, epsilon + 0.1]
i = 1
h = 20
h_increment = 1

while(abs(rmse_values[i] - rmse_values[i - 1]) > epsilon):

    Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

    Y_pred_test = kernel_smoother(X_test, x_t_test, Y_test, h)
    rmse = RMSE(Y_test, Y_pred_test)
    rmse_values.append(rmse)

    h += h_increment
    i += 1

Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

# Calculating RMSE:
Y_pred_test = kernel_smoother(X_test, x_t_test, Y_test, h)
rmse = RMSE(Y_test, Y_pred_test)

print("The RMSE for the kernel smoother with the Insurance dataset is {} with h = {}".format(rmse, h))

# Calculating Squared Errors:
from RegressionUtils import mean_squared_errors

x_t_test = linspace(min(X_test), max(X_test), len(X_test))
Y_pred_test = kernel_smoother(X_test, x_t_test, Y_test, h)
mse = mean_squared_errors(Y_test, Y_pred_test)

print("The squared errors for the kernel smoother with the Insurance dataset is {}\n".format(mse))

# Printing selected features:
print('Selected features for Insurance data set:')
[print("{}) {}".format(i + 1, feature)) for i, feature in zip(range(len(features)),features)]
print("")

#
# -----------------------------------------------------
# Plotting insurance:

plt.plot(X_train, Y_train, "b.", markersize = 7)
plt.plot(X_test, Y_test, "g.", markersize = 5)

plt.plot(x_t, Y_pred, "r-")

plt.title("Insurance")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Plotting h optimization.
x = linspace(0, len(rmse_values), len(rmse_values))

plt.plot(x, rmse_values, "r-")

plt.title("Error Function")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

#--------------------------------------------------------------------------------

# CarPrice Data retrieval:
file_name = "CarPrice_Assignment.csv"
folder_name = "Datasets"
path = os.path.abspath(os.getcwd()) + '\\' + folder_name + '\\' + file_name
car_price = pd.read_csv(path).to_numpy()

X = np.array(car_price[:: -1])
Y_truth = np.array(car_price[:,-1]).reshape(-1 , 1)

features = pd.read_csv(path).columns.to_numpy()

X = replace_nan_values(X)

# -----------------------------------------------------
# CarPrice Data Manipulation:

# Feartue selection:
from RegressionUtils import select_n_features
# Selecting top 3 features.
n = 3
# car_price = replace_nan_values(car_price)

X, feature_indecies = select_n_features(car_price, n)

# Retrieving the remaining features.
features = features[feature_indecies]

# # Dimensionality reduction:
from RegressionUtils import dimensinality_reduction
# The number of dimensions after performing dimensionality
# reduction
n = 1
X = dimensinality_reduction(X, n)

# # Constructing train and test sets:
from sklearn.model_selection import train_test_split
test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_truth, test_size = test_size, random_state=1)

X_train = np.array(X_train).reshape(1, -1)[0]
X_test = np.array(X_test).reshape(1, -1)[0]
Y_train = np.array(Y_train).reshape(1, -1)[0]
Y_test = np.array(Y_test).reshape(1, -1)[0]

#
# -----------------------------------------------------
# Regression:

from NonparametricRegression import kernel_smoother
x_t = linspace(min(X_train), max(X_train), len(X_train))

x_t_test = linspace(min(X_test), max(X_test), len(X_test))
from RegressionUtils import RMSE

# Optimizing the h hyperparameter:
epsilon = 0.001
rmse_values = [0, epsilon + 0.1]
i = 1
h = 20
h_increment = 1

while(abs(rmse_values[i] - rmse_values[i - 1]) > epsilon):

    Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

    Y_pred_test = kernel_smoother(X_test, x_t_test, Y_test, h)
    rmse = RMSE(Y_test, Y_pred_test)
    rmse_values.append(rmse)

    h += h_increment
    i += 1

# Regression with optimal h:
Y_pred = kernel_smoother(X_train, x_t, Y_train, h)

# Calculating RMSE:
from RegressionUtils import RMSE
x_t_test = linspace(min(X_test), max(X_test), len(X_test))
Y_pred_test = kernel_smoother(X_test, x_t_test, Y_test, h)
rmse = RMSE(Y_test, Y_pred_test)

print("The RMSE for the kernel smoother with the CarPrice dataset is {} with h = {}".format(rmse, h))

# Calculating Squared Errors:
from RegressionUtils import mean_squared_errors

x_t_test = linspace(min(X_test), max(X_test), len(X_test))
Y_pred_test = kernel_smoother(X_test, x_t_test, Y_test, h)
mse = mean_squared_errors(Y_test, Y_pred_test)

print("The squared errors for the kernel smoother with the CarPrice dataset is {}\n".format(mse))

# Printing selected features:
print('Selected features in Car Price data set:')
[print("{}) {}".format(i + 1, feature)) for i, feature in zip(range(len(features)),features)]
print("")

#
# -----------------------------------------------------
# Plotting CarPrice:

plt.plot(X_train, Y_train, "b.", markersize = 5)
plt.plot(X_test, Y_test, "g.", markersize = 5)

plt.plot(x_t, Y_pred, "r-")

plt.title("Car Price")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Plotting h optimization.
x = linspace(0, len(rmse_values), len(rmse_values))

plt.plot(x, rmse_values, "r-")

plt.title("Error Function")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# -----------------------------------------------------