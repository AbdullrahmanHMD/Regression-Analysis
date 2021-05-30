import numpy as np
import math


# RMSE: Root Mean Squared Error.
# Y:        The true labels of a data set.
# Y_pred:   The predicted labels of a data set.
# N:        The number of points in the data set
#           whose labels are Y.
def RMSE(Y, Y_pred, N):
    delta_Y = np.subtract(Y, Y_pred)
    delta_Y = np.square(delta_Y)
    delta_Y = np.sum(delta_Y)

    rmse = np.divide(delta_Y, N)
    rmse = math.sqrt(rmse)

    return rmse

def split_data(X, Y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
    return X_train, X_test, Y_train, Y_test

# select_n_features: Given a dataset and an n value
# returns the dataset with the n most impactful
# features.
# dataset:  The dataset to select features from.
# n:        The number of features to select.
def select_n_features(dataset, n):

    # If the first parameter is not an array.
    if not (isinstance(dataset, list) or isinstance(dataset, np.ndarray)):
        raise Exception("Invalid input type. X is of type {}".format(type(dataset)))

    # If the n parameter is geater than the number of feautres of the given dataset.
    if n > dataset.shape[1]:
        raise Exception("Invalid n value for a dataset with {} features.".format(dataset.shape[1]))

    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest

    Y_truth = np.array(dataset[:,-1])
    X = np.array(dataset[:,1: -1])

    selected = SelectKBest(score_func=chi2, k='all')
    fit = selected.fit(X, Y_truth.astype('int'))
    scores = np.array(fit.scores_)

    selected_scores = scores
    
    for i in range(len(scores) - n):
        selected_scores = np.delete(selected_scores, np.argmin(selected_scores))

    selected_scores_indices = []

    for s in selected_scores:
        selected_scores_indices.append(list(scores).index(s))

    return np.array([np.stack((X[:,i]), axis=0) for i in selected_scores_indices]).T


# dimensinality_reduction: Given a list of data points X
# reduces their dimensionality to a given parameter n.
# X: The data points to perform diemsionality reduction upon.
# n: The dimension of the data points after performing
#    dimensionality reduction.   
def dimensinality_reduction(X, n):
  from sklearn.decomposition import PCA 
  model = PCA(n_components=n) 
  model.fit(X) 
  X = model.transform(X)
  return X