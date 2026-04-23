import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class SimpleKMedoids:
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        # Randomly select initial medoids indices
        self.medoid_indices_ = np.random.choice(len(X), self.n_clusters, replace=False)
        self.medoids_ = X[self.medoid_indices_]
        
        for _ in range(self.max_iter):
            distances = pairwise_distances(X, self.medoids_)
            self.labels_ = np.argmin(distances, axis=1)
            
            new_medoid_indices = np.zeros(self.n_clusters, dtype=int)
            for i in range(self.n_clusters):
                cluster_points_indices = np.where(self.labels_ == i)[0]
                if len(cluster_points_indices) == 0:
                    new_medoid_indices[i] = self.medoid_indices_[i]
                    continue
                cluster_points = X[cluster_points_indices]
                intra_distances = pairwise_distances(cluster_points, cluster_points)
                sum_distances = intra_distances.sum(axis=1)
                best_medoid_idx = cluster_points_indices[np.argmin(sum_distances)]
                new_medoid_indices[i] = best_medoid_idx
                
            if np.array_equal(self.medoid_indices_, new_medoid_indices):
                break
            self.medoid_indices_ = new_medoid_indices
            self.medoids_ = X[self.medoid_indices_]
            
        self.inertia_ = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels_ == i]
            if len(cluster_points) > 0:
                self.inertia_ += np.sum(pairwise_distances(cluster_points, [self.medoids_[i]]))
                
        return self

    def predict(self, X):
        distances = pairwise_distances(X, self.medoids_)
        return np.argmin(distances, axis=1)
