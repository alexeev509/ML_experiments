import pandas as pd
import numpy as np
import implicit
from scipy import sparse as sp

# user_rating_df = pd.read_csv(r'C:\Users\alexe\OneDrive\Рабочий стол\ANIME\user_ratings.csv')
# anime_df = pd.read_csv(r'C:\Users\alexe\OneDrive\Рабочий стол\ANIME\anime.csv')
# print(user_rating_df.head(100))

# for i in user_rating_df.index:
#     print(user_rating_df.loc[i,'user_id'])





# print(user_rating_df.loc[19082568,'user_id']) =98598
#test_df=pd.read_csv(r'C:\Users\alexe\OneDrive\Рабочий стол\ANIME\test.csv')

#print(test_df.head(3))




example = pd.DataFrame(np.arange(12).reshape(3,4),columns=['a','b','c','d'])
example.at[1, 'b'] = 0
example.at[0, 'c'] = 0
example.at[2, 'c'] = 0
example.at[1, 'd'] = 0
print(example.head(4))

dense_matrix = example.to_numpy()

print(dense_matrix)

dense_matrix=dense_matrix*0.1
print("we multiply matrix on 0.1")
print(dense_matrix)

result_sparce_matrix=sp.coo_matrix(dense_matrix).tocsr()
print(result_sparce_matrix)

model = implicit.als.AlternatingLeastSquares(factors=16, regularization =0.01, iterations = 8)

model.fit(result_sparce_matrix.T)

query_sparce_matrix=sp.coo_matrix(np.array([[4,0,6,0]])).tocsr()

print(model.recommend(0,query_sparce_matrix,N=4))


sparse_matrix = sp.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
sparse_matrix2 = sp.csc_matrix(np.array([[0, 5, 3], [4, 0, 0]]))
sparse_matrix3 = sp.csc_matrix(np.array([[4, 5, 3]]))
sparse_matrix4 = sp.csc_matrix(np.array([[9, 8, 99]]))

sp.save_npz('f1.npz', sparse_matrix)

print(sp.load_npz("f1.npz"))

print("UNION 1")
print(sp.vstack([sparse_matrix,sparse_matrix2]))
result = sp.vstack([sparse_matrix,sparse_matrix2])

result=sp.vstack([result,sparse_matrix3])
print("UNION 2")
print(result)

sp.save_npz('f2.npz', result)
sp.save_npz('f3.npz', sparse_matrix4)

print("UNION 2")
print(sp.vstack([sp.load_npz('f2.npz'),sp.load_npz('f3.npz')]))
