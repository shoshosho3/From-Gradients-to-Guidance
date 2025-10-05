import numpy as np
import pandas as pd

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
    """
    Numerically stable sigmoid function.
    :param z: Input array
    :return: Sigmoid of the input
    """
    return 1.0 / (1.0 + np.exp(-z))

def sample_group(n, mu, Sigma, rng):
    """
    Samples n points from a multivariate normal distribution.
    :param n: Number of samples
    :param mu: Mean vector
    :param Sigma: Covariance matrix
    :param rng: Random number generator
    :return: Sampled points of shape (n, len(mu))
    """
    return rng.multivariate_normal(mu, Sigma, size=n)

def make_labels(X, w_star, rng):
    """
    Generates binary labels using a logistic model.
    :param X: Input features of shape (n_samples, n_features)
    :param w_star: True weight vector
    :param rng: Random number generator
    :return: Binary labels of shape (n_samples,)
    """
    p = sigmoid(X @ w_star)
    return rng.binomial(1, p, size=len(X)).astype(float)

def train_logreg(X, y, steps=STEPS, lr=LR, lam=L2):
    """
    Trains a logistic regression model using gradient descent.
    :param X: Input features of shape (n_samples, n_features)
    :param y: Binary labels of shape (n_samples,)
    :param steps: Number of gradient descent steps
    :param lr: Learning rate
    :param lam: L2 regularization strength
    :return: Learned weight vector of shape (n_features,)
    """
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        z = X @ w
        p = sigmoid(z)
        grad = (X.T @ (p - y)) / len(y) + lam*w
        w -= lr * grad
    return w

def entropy_scores(X, w):
    """
    Binary CE entropy scores for linear logit z = w^T x
    :param X: Input features of shape (n_samples, n_features)
    :param w: Weight vector of shape (n_features,)
    :return: Entropy scores of shape (n_samples,)
    """
    p = sigmoid(X @ w)
    eps = 1e-12
    return -(p*np.log(p+eps) + (1-p)*np.log(1-p+eps))

def egl_scores(X, w):
    """
    Binary CE exact EGL for linear logit z = w^T x
    :param X: Input features of shape (n_samples, n_features)
    :param w: Weight vector of shape (n_features,)
    :return: EGL scores of shape (n_samples,)
    """
    p = sigmoid(X @ w)
    return 2.0 * p * (1.0 - p) * np.linalg.norm(X, axis=1)

def eval_group_acc(w, mu, Sigma, w_star, rng, n_test=N_TEST):
    """
    Evaluates accuracy on a sampled test set from a group.
    :param w: Weight vector of the model to evaluate
    :param mu: Mean vector of the group
    :param Sigma: Covariance matrix of the group
    :param w_star: True weight vector for generating labels
    :param rng: Random number generator
    :param n_test: Number of test samples
    :return: Accuracy on the test set
    """
    X = sample_group(n_test, mu, Sigma, rng)
    y_true = make_labels(X, w_star, rng)
    y_pred = (sigmoid(X @ w) >= 0.5).astype(float)
    return (y_true == y_pred).mean()

def init_experiment(seed):
    """
    This function initializes the experiment parameters and random number generator.
    :param seed: Random seed for reproducibility
    :return: Tuple of (rng, mu1, mu2, Sigma, w_star)
    """
    rng = np.random.default_rng(seed)
    mu1 = np.zeros(D); mu1[0] = MU1
    mu2 = np.zeros(D); mu2[0] = MU2
    Sigma = np.eye(D) * SIGMA_SCALE
    w_star = np.zeros(D); w_star[0] = 1.0  # teacher
    return rng, mu1, mu2, Sigma, w_star


def create_labeled_set(mu1, mu2, Sigma, w_star, rng):
    """
    Creates the initial labeled dataset and trains the initial model.
    :param mu1: Mean vector for group S1
    :param mu2: Mean vector for group S2
    :param Sigma: Shared covariance matrix
    :param w_star: True weight vector for generating labels
    :param rng: Random number generator
    :return: Tuple of (X_L, y_L, w0) where X_L is the labeled features,
             y_L are the labels, and w0 is the trained weight vector
    """
    X_L = np.vstack([
        sample_group(M1, mu1, Sigma, rng),
        sample_group(M2, mu2, Sigma, rng),
    ])
    y_L = make_labels(X_L, w_star, rng)
    w0 = train_logreg(X_L, y_L)
    return X_L, y_L, w0


def evaluate_pre_query_acc(w0, mu1, mu2, Sigma, w_star, rng):
    """
    Evaluates pre-query accuracies on both groups.
    :param w0: Weight vector of the initial model
    :param mu1: Mean vector for group S1
    :param mu2: Mean vector for group S2
    :param Sigma: Shared covariance matrix
    :param w_star: True weight vector for generating labels
    :param rng: Random number generator
    :return: Tuple of (acc_S1, acc_S2) accuracies on groups S1 and S2
    """
    acc_S1 = eval_group_acc(w0, mu1, Sigma, w_star, rng)
    acc_S2 = eval_group_acc(w0, mu2, Sigma, w_star, rng)
    return acc_S1, acc_S2


def create_unlabeled_pool(mu1, mu2, Sigma, rng):
    """
    Creates the unlabeled pool dataset.
    :param mu1: Mean vector for group S1
    :param mu2: Mean vector for group S2
    :param Sigma: Shared covariance matrix
    :param rng: Random number generator
    :return: Tuple of (poolX, poolG) where poolX are the features
             and poolG are the group labels (0 for S1, 1 for S2)
    """
    pool_S1 = sample_group(N1, mu1, Sigma, rng)
    pool_S2 = sample_group(N2, mu2, Sigma, rng)
    poolX = np.vstack([pool_S1, pool_S2])
    poolG = np.array([0]*len(pool_S1) + [1]*len(pool_S2))  # 0=S1, 1=S2
    return poolX, poolG


def compute_scores(poolX, w0):
    """
    Computes entropy and EGL scores for the unlabeled pool.
    :param poolX: Unlabeled features of shape (n_samples, n_features)
    :param w0: Weight vector of the initial model
    :return: Tuple of (H, G) where H are entropy scores and G are EGL scores
    """
    H = entropy_scores(poolX, w0)
    G = egl_scores(poolX, w0)
    return H, G


def select_candidates(poolX, poolG, H, G, rng):
    """
    Selects candidates based on random, entropy, and EGL strategies.
    :param poolX: Unlabeled features of shape (n_samples, n_features)
    :param poolG: Group labels of shape (n_samples,)
    :param H: Entropy scores of shape (n_samples,)
    :param G: EGL scores of shape (n_samples,)
    :param rng: Random number generator
    :return: Tuple of indices and next group indicators for each strategy
    (idx_rand, idx_ent, idx_egl, next_S2_rand, next_S2_ent, next_S2_egl)
    1 if the selected sample is from S2, else 0
    """
    idx_rand = int(rng.integers(0, len(poolX)))
    idx_ent  = int(np.argmax(H))
    idx_egl  = int(np.argmax(G))
    next_S2_rand = int(poolG[idx_rand] == 1)
    next_S2_ent  = int(poolG[idx_ent]  == 1)
    next_S2_egl  = int(poolG[idx_egl]  == 1)
    return (idx_rand, idx_ent, idx_egl,
            next_S2_rand, next_S2_ent, next_S2_egl)


def delta_S2(idx, X_L, y_L, poolX, mu2, Sigma, w_star, rng, acc_S2):
    """
    Computes the change in S2 accuracy after adding one queried sample.
    :param idx: Index of the queried sample in poolX
    :param X_L: Current labeled features
    :param y_L: Current labeled labels
    :param poolX: Unlabeled features of shape (n_samples, n_features)
    :param mu2: Mean vector for group S2
    :param Sigma: Shared covariance matrix
    :param w_star: True weight vector for generating labels
    :param rng: Random number generator
    :param acc_S2: Current accuracy on group S2
    :return: Change in accuracy on group S2 after adding the new sample
    """
    X_new = poolX[idx:idx+1]
    y_new = make_labels(X_new, w_star, rng)
    X_aug = np.vstack([X_L, X_new])
    y_aug = np.concatenate([y_L, y_new])
    w1 = train_logreg(X_aug, y_aug, steps=STEPS//2)
    acc_after = eval_group_acc(w1, mu2, Sigma, w_star, rng)
    return acc_after - acc_S2


def one_trial(seed):
    """
    Runs one trial of the experiment with a given random seed.
    :param seed: Random seed for reproducibility
    :return: Dictionary of results from the trial
    """
    # Initialization
    rng, mu1, mu2, Sigma, w_star = init_experiment(seed)

    # Phase 1: labeled data
    X_L, y_L, w0 = create_labeled_set(mu1, mu2, Sigma, w_star, rng)

    # Pre-query accuracies
    acc_S1, acc_S2 = evaluate_pre_query_acc(w0, mu1, mu2, Sigma, w_star, rng)

    # Phase 2: unlabeled pool
    poolX, poolG = create_unlabeled_pool(mu1, mu2, Sigma, rng)

    # Scores and candidate selection
    H, G = compute_scores(poolX, w0)
    (idx_rand, idx_ent, idx_egl,
     next_S2_rand, next_S2_ent, next_S2_egl) = select_candidates(poolX, poolG, H, G, rng)

    # Compute ΔS2
    dS2_rand = delta_S2(idx_rand, X_L, y_L, poolX, mu2, Sigma, w_star, rng, acc_S2)
    dS2_ent  = delta_S2(idx_ent,  X_L, y_L, poolX, mu2, Sigma, w_star, rng, acc_S2)
    dS2_egl  = delta_S2(idx_egl,  X_L, y_L, poolX, mu2, Sigma, w_star, rng, acc_S2)

    # Return results
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


def main():
    """Run multiple experiment trials and summarize the results."""
    print(f"Running {len(SEEDS)} trials...\n")

    # --- Run all trials ---
    rows = [one_trial(seed) for seed in SEEDS]
    if not rows:
        print("No trials were run. Check SEEDS configuration.")
        return

    df = pd.DataFrame(rows)

    # --- Summary statistics ---
    summary = df.mean(numeric_only=True).to_dict()
    summary["trials"] = len(df)

    # Reorder columns for readability (optional)
    ordered_keys = [
        "trials",
        "acc_S1_before",
        "acc_S2_before",
        "P(next∈S2|Random)",
        "P(next∈S2|Entropy)",
        "P(next∈S2|EGL)",
        "ΔS2|Random",
        "ΔS2|Entropy",
        "ΔS2|EGL",
    ]
    summary = {k: summary[k] for k in ordered_keys if k in summary}

    print("=== Summary (Mean Across Seeds) ===")
    print(pd.Series(summary).round(4).to_string())
    print()

if __name__ == "__main__":
    main()
