import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from .strategy import Strategy

class BaseDiversityStrategy(Strategy):
    """
    A base class for diversity-aware query strategies.

    Child classes must implement the `_get_scores_and_embeddings` method to
    specify how to obtain scores and embeddings from their specific network handler.
    """

    def __init__(self, dataset, net, args_input, args_task):
        super().__init__(dataset, net, args_input, args_task)
        # factoring to determine the size of the candidate pool
        self.candidate_factor = getattr(self.args_input, "candidate_factor", 5.0)

    def _get_scores_and_embeddings(self, unlabeled_data):
        """
        Abstract method to be implemented by child classes.
        Should return scores and embeddings for the given data.

        :param unlabeled_data: The dataset of unlabeled samples.
        :return: A tuple of (scores, embeddings), both as numpy arrays.
        """
        raise NotImplementedError

    def _select_candidates(self, scores, num_candidates):
        """
        This method exists for reproducibility purposes. It is overridable by child classes.
        """
        return scores.argsort()[-num_candidates:]

    def _get_candidate_pool(self, unlabeled_idxs, scores, embeddings, n):
        """
        Filters the top candidates based on scores to form a candidate pool.
        :param unlabeled_idxs: Indices of the unlabeled samples.
        :param scores: Scores indicating informativeness of samples.
        :param embeddings: Feature embeddings of the samples.
        :param n: Number of samples to query.
        :return: A tuple of (candidate_embeddings, candidate_global_idxs).
        """
        num_candidates = min(len(scores), int(self.candidate_factor * n))
        candidate_local_indices = self._select_candidates(scores, num_candidates)
        candidate_embeddings = embeddings[candidate_local_indices]
        candidate_global_idxs = unlabeled_idxs[candidate_local_indices]
        return candidate_embeddings, candidate_global_idxs

    def _cluster_embeddings(self, candidate_embeddings, n):
        """
        Clusters the candidate embeddings into n clusters using K-Means.
        :param candidate_embeddings: Embeddings of the candidate samples.
        :param n: Number of clusters (final samples to select).
        :return: KMeans object after fitting.
        """
        kmeans = KMeans(n_clusters=n, n_init=10, random_state=0)
        kmeans.fit(candidate_embeddings)
        return kmeans

    def _select_diverse_samples(self, kmeans, candidate_embeddings, candidate_global_idxs):
        """
        Selects the most representative sample from each cluster.
        :param kmeans: Fitted KMeans object.
        :param candidate_embeddings: Embeddings of the candidate samples.
        :param candidate_global_idxs: Global indices of the candidate samples.
        :return: Indices of the selected samples.
        """
        distances = euclidean_distances(kmeans.cluster_centers_, candidate_embeddings)
        final_candidate_indices = np.argmin(distances, axis=1)
        return candidate_global_idxs[final_candidate_indices]

    def query(self, n):
        """
        Queries n diverse and informative samples.
        :param n: Number of samples to query.
        :return: Indices of the selected samples.
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        scores, embeddings = self._get_scores_and_embeddings(unlabeled_data)
        candidate_embeddings, candidate_global_idxs = self._get_candidate_pool(unlabeled_idxs, scores, embeddings, n)
        kmeans = self._cluster_embeddings(candidate_embeddings, n)
        final_indices = self._select_diverse_samples(kmeans, candidate_embeddings, candidate_global_idxs)
        return final_indices

