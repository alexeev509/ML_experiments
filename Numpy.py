import pandas as pd
import numpy as np

user_rating_df = pd.read_csv(r'C:\Users\alexe\OneDrive\Рабочий стол\ANIME\user_ratings.csv')
anime_df = pd.read_csv(r'C:\Users\alexe\OneDrive\Рабочий стол\ANIME\anime.csv')
print(anime_df['Id'].value_counts())

numRows = 98600
numCols = 15072
# temp =pd.DataFrame(index=range(numRows),columns=range(numCols))
s = (numRows, numCols)
temp = np.zeros(s, dtype='int64')
print(temp)
row, col = temp.shape
print("Rows count: ",row)
# 3
print("Cols count: ",col)
