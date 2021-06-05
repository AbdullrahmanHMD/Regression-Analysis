import numpy as np
import pandas as pd
  
# reading the csv file
df = pd.read_csv("CarPrice_Assignment.csv")
aux = df.to_numpy  

col = 'doornumber'

for i in range(len(df)):
    if df.loc[i, col] == 'two':
        df.loc[i, col] = 0
    else:
        df.loc[i, col] = 1

col = 'drivewheel'


for i in range(len(df)):
    if df.loc[i, col] == 'fwd':
        df.loc[i, col] = 0
    else:
        df.loc[i, col] = 1

col = 'fueltype'


for i in range(len(df)):
    if df.loc[i, col] == 'gas':
        df.loc[i, col] = 0
    else:
        df.loc[i, col] = 1

col = 'aspiration'

for i in range(len(df)):
    if df.loc[i, col] == 'std':
        df.loc[i, col] = 0
    else:
        df.loc[i, col] = 1

df.to_csv("CarPrice_Assignment_new.csv", index=False)