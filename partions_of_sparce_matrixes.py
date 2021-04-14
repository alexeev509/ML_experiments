import pandas as pd
import numpy as np
from scipy import sparse as sp

user_rating_df = pd.read_csv(r'C:\Users\alexe\OneDrive\Рабочий стол\ANIME\user_ratings.csv')
anime_df = pd.read_csv(r'C:\Users\alexe\OneDrive\Рабочий стол\ANIME\anime.csv')
print(user_rating_df.info())
print("RESULT: ", user_rating_df.loc[19082568, 'user_id'])
print(anime_df['Id'].value_counts())

print(anime_df.tail(1))  # to see last id of anime (48492)
print(user_rating_df.tail(1))  # to see last id of user (98598)
# we will work with partitions: 98600/4=24650 (really we have 98599 rows)
step = 24650
numRows = 24650
numCols = 48493

s = (numRows, numCols)
temp = np.zeros(s)
print("Rows count: ", numRows, "\nCols count: ", numCols)

# for i in user_rating_df.index:
#    temp[user_rating_df.loc[i,'user_id']][user_rating_df.loc[i,'anime_id']]=user_rating_df.loc[i,'rating']/10
left = 0
right = 19082569
for j in range(0, 4):
    temp = np.zeros(s)
    q = j * step
    # for i in range(q, (j + 1) * step):
    right_index = (j + 1) * step
    for i in pd.RangeIndex(left, right, 1):
        if (user_rating_df.loc[i, 'user_id'] < right_index):
            temp[user_rating_df.loc[i, 'user_id'] % step][user_rating_df.loc[i, 'anime_id']] = user_rating_df.loc[i, 'rating'] / 10
        else:
            left = i
            break
    sp.save_npz("table" + str(j), sp.csc_matrix(temp))


