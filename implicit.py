import pandas as pd
import numpy as np
import implicit
from scipy import sparse as sp



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
