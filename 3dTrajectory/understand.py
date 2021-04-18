import pandas as pd

data = pd.read_csv("nba.csv") 

# retrieving rows by loc method 
print(data.iloc[1:5]) 
#print(row1)
# # retrieving rows by iloc method 
# row2 = data.iloc[3] 

# # checking if values are equal 
# row1 == row2 
