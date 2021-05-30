
import pandas as pd
  
# reading the csv file
df = pd.read_csv("insurance.csv")
aux = df.to_numpy  

col = 'region'

for i in range(len(df)):
    if df.loc[i, col] == 'southwest':
        df.loc[i, col] = 0
    else:
        df.loc[i, col] = 1

# updating the column value/data
# writing into the file
df.to_csv("insurance.csv", index=False)