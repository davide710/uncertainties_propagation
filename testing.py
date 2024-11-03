import numpy as np
import pandas as pd
from propagation import propagate

df = pd.read_csv('test_data/refraction_water.csv')

h = 14.7
sh = 0.005

n_points = df.shape[0]

df['y'] = np.sin(np.arctan2(df.l_air, h))

f = lambda X: np.sin(np.arctan2(X[0], X[1]))

X = [[df.l_air[i], h] for i in range(n_points)]
sigma = [[df.l_error[i], sh] for i in range(n_points)]

y, sy = propagate(X, sigma, f)

print(f'Percentual differences: \n{(sy - df.y_error_with_derivatives) / df.y_error_with_derivatives * 100}')
