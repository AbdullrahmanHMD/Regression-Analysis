# **Regression-Analysis:**

## Description:
  Regression Analysis is a project where we tested three different regression functions on three different data sets and compared the overall performance of the regression         functions.

  The regression functions being: Linear regression, polynomial regression and nonparametric regression.

## Date Retrieval:
  We used three different data sets which are: Real estate, insurance and car price data sets retrieved from Kaggle website as csv files. We then imported them to our workspace in python.
The data sets are located in a folder called Datasets in the project files.

## Feature Manipulation:
   - **Feature extraction:**
      Using the CHI2 function from Sklearn library we picked the most impactful features on the Y values of the data set and did the modeling according to them.

  - **Dimensionality Reduction:**
    We used PCA from Sklearn library to perform the dimensionality reduction on the remaining features. For all the data sets, we reduced the dimensionality to 1 in order to be able to plot the regression function in 2 dimensions so we could have better interpretation on the data.

## References:
  - Ethem Alpaydin - Introduction to Machine Learning (2014, The MIT Press)
