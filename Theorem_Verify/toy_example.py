import numpy as np

# ----------------------------- Config -----------------------------
D = 6
# Labeled-set sizes (heavily S1-biased)
M1, M2 = 600, 4
# Unlabeled pool (set to (8000, 400) for heavy skew; or (1500, 1500) for balanced)
N1, N2 = 8000, 400
# Geometry: means along e1, shared spherical covariance
MU1, MU2 = +2.5, -0.4
SIGMA_SCALE = 0.3
# Training hyperparams
STEPS, LR, L2 = 240, 0.25, 1e-2
# Evaluation
N_TEST = 2000
SEEDS = range(1000)
# ------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sample_group(n, mu, Sigma, rng):
    return rng.multivariate_normal(mu, Sigma, size=n)

def make_labels(X, w_star, rng):
    p = sigmoid(X @ w_star)
    return rng.binomial(1, p, size=len(X)).astype(float)

def train_logreg(X, y, steps=STEPS, lr=LR, lam=L2):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        z = X @ w
        p = sigmoid(z)
        grad = (X.T @ (p - y)) / len(y) + lam*w
        w -= lr * grad
    return w

def entropy_scores(X, w):
    p = sigmoid(X @ w)
    eps = 1e-12
    return -(p*np.log(p+eps) + (1-p)*np.log(1-p+eps))

def egl_scores(X, w):
    # Binary CE exact EGL for linear logit z = w^T x
    p = sigmoid(X @ w)
    return 2.0 * p * (1.0 - p) * np.linalg.norm(X, axis=1)

def eval_group_acc(w, mu, Sigma, w_star, rng, n_test=N_TEST):
    X = sample_group(n_test, mu, Sigma, rng)
    y_true = make_labels(X, w_star, rng)
    y_pred = (sigmoid(X @ w) >= 0.5).astype(float)
    return (y_true == y_pred).mean()

def one_trial(seed):
    rng = np.random.default_rng(seed)
    # Data geometry
    mu1 = np.zeros(D); mu1[0] = MU1
    mu2 = np.zeros(D); mu2[0] = MU2
    Sigma = np.eye(D) * SIGMA_SCALE
    w_star = np.zeros(D); w_star[0] = 1.0  # teacher

    # Phase 1: S1-heavy labeled set
    X_L = np.vstack([
        sample_group(M1, mu1, Sigma, rng),
        sample_group(M2, mu2, Sigma, rng),
    ])
    y_L = make_labels(X_L, w_star, rng)
    w0 = train_logreg(X_L, y_L)

    # Pre-query accuracies (diagnostic)
    acc_S1 = eval_group_acc(w0, mu1, Sigma, w_star, rng)
    acc_S2 = eval_group_acc(w0, mu2, Sigma, w_star, rng)

    # Phase 2: equal or skewed unlabeled pool
    pool_S1 = sample_group(N1, mu1, Sigma, rng)
    pool_S2 = sample_group(N2, mu2, Sigma, rng)
    poolX = np.vstack([pool_S1, pool_S2])
    poolG = np.array([0]*len(pool_S1) + [1]*len(pool_S2))  # 0=S1, 1=S2

    # Scores on the frozen model
    H = entropy_scores(poolX, w0)
    G = egl_scores(poolX, w0)

    # Top-1 indices and labels
    idx_rand = int(rng.integers(0, len(poolX)))
    idx_ent  = int(np.argmax(H))
    idx_egl  = int(np.argmax(G))
    next_S2_rand = int(poolG[idx_rand] == 1)
    next_S2_ent  = int(poolG[idx_ent]  == 1)
    next_S2_egl  = int(poolG[idx_egl]  == 1)

    # One-step S2 accuracy delta after adding that single point
    def delta_S2(idx):
        X_new = poolX[idx:idx+1]
        y_new = make_labels(X_new, w_star, rng)
        X_aug = np.vstack([X_L, X_new])
        y_aug = np.concatenate([y_L, y_new])
        w1 = train_logreg(X_aug, y_aug, steps=STEPS//2)
        acc_after = eval_group_acc(w1, mu2, Sigma, w_star, rng)
        return acc_after - acc_S2

    dS2_rand = delta_S2(idx_rand)
    dS2_ent  = delta_S2(idx_ent)
    dS2_egl  = delta_S2(idx_egl)

    return {
        "seed": seed,
        "acc_S1_before": acc_S1,
        "acc_S2_before": acc_S2,
        "P(next∈S2|Random)": next_S2_rand,
        "P(next∈S2|Entropy)": next_S2_ent,
        "P(next∈S2|EGL)":    next_S2_egl,
        "ΔS2|Random": dS2_rand,
        "ΔS2|Entropy": dS2_ent,
        "ΔS2|EGL":    dS2_egl,
    }

# Run trials
rows = [one_trial(s) for s in SEEDS]
import pandas as pd
df = pd.DataFrame(rows)
print(df.round(4))

print("\nSummary (means across seeds):")
summary = {
    "trials": len(df),
    "S1 acc before": df["acc_S1_before"].mean(),
    "S2 acc before": df["acc_S2_before"].mean(),
    "P(next∈S2)|Random": df["P(next∈S2|Random)"].mean(),
    "P(next∈S2)|Entropy": df["P(next∈S2|Entropy)"].mean(),
    "P(next∈S2)|EGL":    df["P(next∈S2|EGL)"].mean(),
    "ΔS2|Random": df["ΔS2|Random"].mean(),
    "ΔS2|Entropy": df["ΔS2|Entropy"].mean(),
    "ΔS2|EGL":    df["ΔS2|EGL"].mean(),
}
print(pd.Series(summary).round(4))
