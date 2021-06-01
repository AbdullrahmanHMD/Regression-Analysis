import numpy as np
import pandas as pd
  
# reading the csv file
df = pd.read_csv("Life_Expectancy_Data.csv")
aux = df.to_numpy  

col = 'Alcohol'

aux = df[col].to_numpy()
column = df[col].to_numpy()

nan_array = np.isnan(aux)
not_nan_array = ~ nan_array
aux = aux[not_nan_array]
avg = np.average(aux)

column[nan_array] = avg

print(aux.shape, " | ", column.shape)

for i in column:
    print(i)

# for i in range(len(df)):
#     if df.loc[i, col] == None:
    
# updating the column value/data
# writing into the file
# df.to_csv("Life_Expectancy_Data_New.csv", index=False)