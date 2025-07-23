import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 1) Load & preprocess Fashion-MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1).numpy()),  # flatten to 784-d
])
full_train = FashionMNIST(root="~/data", train=True, download=True, transform=transform)
test_ds = FashionMNIST(root="~/data", train=False, download=True, transform=transform)

# We'll take a random 10k subset for speed
train_idx = np.random.RandomState(0).choice(len(full_train), size=10000, replace=False)
train_ds = Subset(full_train, train_idx)

X_train = np.stack([full_train[i][0] for i in train_idx])
y_train = np.array([full_train[i][1] for i in train_idx])
X_test = np.stack([x for x, _ in test_ds])
y_test = np.array([y for _, y in test_ds])

# 2) One‑hot encode labels for softmax
num_classes = 10
Y_train_oh = np.eye(num_classes)[y_train]

# 3) Active‐learning + Adam parameters
M = 100  # batch size per iteration (tune this)
β1, β2 = 0.9, 0.999
eps = 1e-8
η = 0.01
budget = X_train.shape[0]
K_batches = budget // M

# 4) Initialize weights & Adam moments for both methods
d, C = X_train.shape[1], num_classes
W_act = np.zeros((d, C))
m_act = np.zeros((d, C))
v_act = np.zeros((d, C))
t_act = 0
W_rnd = np.zeros((d, C))
m_rnd = np.zeros((d, C))
v_rnd = np.zeros((d, C))
t_rnd = 0

selected_act = set()
selected_rnd = set()

perf_act, perf_rnd, labels_used = [], [], []


# helper: one‐step Adam update for softmax‐cross‐entropy
def adam_step(W, m, v, t, x, y_oh):
    t += 1
    # forward + grad (softmax CE)
    logits = x.dot(W)  # (C,)
    expL = np.exp(logits - logits.max())
    p = expL / expL.sum()
    grad = np.outer(x, p - y_oh)  # (d,C)
    # Adam updates
    m = β1 * m + (1 - β1) * grad
    v = β2 * v + (1 - β2) * (grad * grad)
    m_hat = m / (1 - β1 ** t)
    v_hat = v / (1 - β2 ** t)
    W_old = W.copy()
    W -= η * m_hat / (np.sqrt(v_hat) + eps)
    delta = np.linalg.norm(W - W_old)
    return W, m, v, t, delta


# 5) INITIAL BATCH (same for both)
np.random.seed(0)
init = np.random.choice(len(X_train), size=M, replace=False)
for idx in init:
    W_act, m_act, v_act, t_act, δ = adam_step(W_act, m_act, v_act, t_act,
                                              X_train[idx], Y_train_oh[idx])
    W_rnd, m_rnd, v_rnd, t_rnd, _ = adam_step(W_rnd, m_rnd, v_rnd, t_rnd,
                                              X_train[idx], Y_train_oh[idx])
selected_act.update(init)
selected_rnd.update(init)


# record initial performance
def compute_acc(W):
    logits = X_test.dot(W)
    preds  = logits.argmax(axis=1)
    return accuracy_score(y_test, preds)


perf_act.append(compute_acc(W_act))
perf_rnd.append(compute_acc(W_rnd))
labels_used.append(len(selected_act))

# KEEP TRACK OF x* (least‐update example) for active
x_star = X_train[init[np.argmin([
    np.linalg.norm(W_act - (W_act - η * (m_act / (np.sqrt(v_act) + eps))))
    for _ in init])]]

# 6) ITERATE
for _ in range(2, K_batches+1):
    # 1) Build pool of as‑yet unlabeled indices
    pool = np.array(list(set(range(len(X_train))) - selected_act))

    # 2) Compute similarity to x_star
    sims = cosine_similarity(X_train[pool],
                             x_star.reshape(1, -1)).ravel()

    # 3) Find the index of the *closest* point
    closest_rel = np.argmax(sims)  # highest cosine similarity

    # 4) Exclude it, then sample M at random
    eligible = np.delete(pool, closest_rel)
    batch_act = np.random.choice(eligible, size=M, replace=False)
    selected_act.update(batch_act)

    # 5) ALSO sample a random batch for the baseline
    pool_r = list(set(range(len(X_train))) - selected_rnd)
    batch_rnd = np.random.choice(pool_r, size=M, replace=False)
    selected_rnd.update(batch_rnd)

    # 6) Train on each new active batch, *and* track the smallest delta to update x_star
    best_delta = float('inf')
    for idx in batch_act:
        W_act, m_act, v_act, t_act, δ = adam_step(
            W_act, m_act, v_act, t_act,
            X_train[idx], Y_train_oh[idx]
        )
        if δ < best_delta:
            best_delta = δ
            x_star = X_train[idx]

    # 7) Train the random baseline on its batch
    for idx in batch_rnd:
        W_rnd, m_rnd, v_rnd, t_rnd, _ = adam_step(
            W_rnd, m_rnd, v_rnd, t_rnd,
            X_train[idx], Y_train_oh[idx]
        )

    # 8) Record performances
    perf_act.append(compute_acc(W_act))
    perf_rnd.append(compute_acc(W_rnd))
    labels_used.append(len(selected_act))


# 7) PLOT
plt.plot(labels_used, perf_act, label="Proposed Active")
plt.plot(labels_used, perf_rnd, label="Random")
plt.xlabel("Number of Labeled Points")
plt.ylabel("Test Accuracy")
plt.title("Fashion‑MNIST: Active vs Random Sampling")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
