import numpy as np

#TODO: support for pandas dataframe
#TODO: support for not having uncertainty on all values

def propagate(X, errors, f):
    """
    X is of shape (n, m), n datapoints, m variables
    """
    X = np.array(X)
    n, m = X.shape
    errors = np.array(errors)

    y = np.array([f(X[i, :]) for i in range(n)])
    sigma_y = np.zeros_like(y)

    for j in range(m):
        e_j = np.zeros(m)
        e_j[j] = 1

        sigma_j = np.dot(errors, np.outer(e_j, e_j))
        X_plus = X + sigma_j
        X_minus = X - sigma_j

        y_plus = np.array([f(X_plus[i, :]) for i in range(n)])
        y_minus = np.array([f(X_minus[i, :]) for i in range(n)])

        sigma_y = sigma_y + (y_plus - y_minus)**2 / 4
    
    sigma_y = np.sqrt(sigma_y)

    print(y)
    print(sigma_y)
    


f = lambda x: x[0] + x[1]
propagate([[1, 2], [3, 4], [8, 5]], [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]], f)

#f = lambda x: x**2
#propagate([[1], [3], [8]], [[0.1], [0.1], [0.1]], f)
