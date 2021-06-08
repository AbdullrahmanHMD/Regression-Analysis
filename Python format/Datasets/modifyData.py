import numpy as np
import pandas as pd
  
# reading the csv file
file_name = "CarPrice_Assignment.csv"
df = pd.read_csv(file_name)

cn = 'price'
col = df[cn]
col = col.to_numpy()

print("Car Price: ", np.average(col))

file_name = 'Real_estate.csv'
df = pd.read_csv(file_name)

cn = 'house price of unit area'
col = df[cn]

col = col.to_numpy()

print("Real Estate: ",np.average(col))

file_name = 'insurance.csv'
df = pd.read_csv(file_name)

cn = 'charges'
col = df[cn]

col = col.to_numpy()

print("Insurance: ", np.average(col))

# df.to_csv("CarPrice_Assignment_new.csv", index=False)