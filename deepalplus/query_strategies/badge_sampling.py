import numpy as np
import torch
from .strategy import Strategy
from scipy import stats
from sklearn.metrics import pairwise_distances
import pdb
from tqdm import tqdm

'''
This implementation is originated from https://github.com/JordanAsh/badge.
Please cite the original paper if you use this method.
@inproceedings{ash2019deep,
  author    = {Jordan T. Ash and
               Chicheng Zhang and
               Akshay Krishnamurthy and
               John Langford and
               Alekh Agarwal},
  title     = {Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds},
  booktitle = {8th International Conference on Learning Representations, {ICLR} 2020,
               Addis Ababa, Ethiopia, April 26-30, 2020},
  publisher = {OpenReview.net},
  year      = {2020}
}
'''

class BadgeSampling(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(BadgeSampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        gradEmbedding = self.get_grad_embeddings(unlabeled_data)
        chosen = init_centers(gradEmbedding, n)
        return unlabeled_idxs[chosen]

# kmeans ++ initialization
def init_centers(X, K):
    """
    A fast, vectorized implementation of k-means++ initialization.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The data to choose centers from.
    K : int
        The number of centers to choose.

    Returns
    -------
    list
        A list of indices of the chosen centers.
    """
    # Ensure X is a 2D numpy array
    X = np.asarray(X)
    n_samples = X.shape[0]

    # Choose the first center: the point furthest from the origin.
    norms = np.linalg.norm(X, axis=1)
    first_center_idx = np.argmax(norms)

    centers_indices = [first_center_idx]

    # Keep track of the squared distance from each point to its nearest center.
    closest_dist_sq = pairwise_distances(X, X[first_center_idx:first_center_idx + 1], metric='euclidean') ** 2
    closest_dist_sq = closest_dist_sq.ravel()

    # Iteratively choose the remaining K-1 centers.
    for _ in tqdm(range(1, K), desc="Selecting centers with k-means++"):
        # Calculate the sampling probabilities proportional to the squared distances.
        # This is the core of k-means++.
        probs = closest_dist_sq / np.sum(closest_dist_sq)

        next_center_idx = np.random.choice(n_samples, p=probs)
        centers_indices.append(next_center_idx)

        # Calculate distances to the *newest* center.
        dist_to_new_center_sq = pairwise_distances(X, X[next_center_idx:next_center_idx + 1], metric='euclidean') ** 2
        dist_to_new_center_sq = dist_to_new_center_sq.ravel()

        # Update the closest distance for each point by taking the minimum.
        closest_dist_sq = np.minimum(closest_dist_sq, dist_to_new_center_sq)

    return centers_indices
