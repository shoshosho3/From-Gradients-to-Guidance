import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=5, noise=10.0, random_state=1)

# Split into pool and test
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Greedy direct subset selection minimizing ||w_full - w_I||
def greedy_direct_subset(X, y, k):
    w_full = np.linalg.solve(X.T @ X, X.T @ y)
    n, d = X.shape
    I = []
    for _ in range(k):
        best_i, best_err = None, np.inf
        for i in range(n):
            if i in I: continue
            I_plus = I + [i]
            X_I, y_I = X[I_plus], y[I_plus]
            if len(I_plus) < d:
                w_I = np.linalg.pinv(X_I) @ y_I
            else:
                # (X_I^T X_I)^-1 X_I^T Y_I
                w_I = np.linalg.solve(X_I.T @ X_I, X_I.T @ y_I)
            err = np.linalg.norm(w_full - w_I)
            if err < best_err:
                best_err, best_i = err, i
        I.append(best_i)
    return I

k = 20
rng = np.random.RandomState(0)

# Compute greedy selection MSE
idx_greedy = greedy_direct_subset(X_pool, y_pool, k)
model_greedy = LinearRegression().fit(X_pool[idx_greedy], y_pool[idx_greedy])
mse_greedy = np.mean((model_greedy.predict(X_test) - y_test)**2)

# Approximate optimum via random sampling
num_samples = 10000
avg_mse = 0
for _ in range(num_samples):
    idx_rand = rng.choice(len(X_pool), size=k, replace=False)
    model_rand = LinearRegression().fit(X_pool[idx_rand], y_pool[idx_rand])
    mse_rand = np.mean((model_rand.predict(X_test) - y_test)**2)
    avg_mse += mse_rand

# Display results
results_df = pd.DataFrame({
    'Strategy': ['Greedy Direct Subset', f'Avg of {num_samples} Random'],
    'Test MSE': [mse_greedy, avg_mse/num_samples]
})
print(results_df)