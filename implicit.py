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

result_sparce_matrix=sp.coo_matrix(dense_matrix).tocsr()
print(result_sparce_matrix)

model = implicit.als.AlternatingLeastSquares(factors=16, regularization =0.0, iterations = 8)

model.fit(result_sparce_matrix.T)

query_sparce_matrix=sp.coo_matrix(np.array([[0,1,0,3]])).tocsr()

print(model.recommend(0,query_sparce_matrix,N=4))
