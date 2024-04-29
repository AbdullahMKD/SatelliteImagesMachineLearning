import numpy as np
import sklearn.metrics as metrics
from sklearn.cluster import KMeans


class kMeans_processing:
    def __init__(self,):
        self.clustered_img = None
        self.clustered_labels = None
        self.inertia = None
        self.silhouette_score = None
        self.centers = None

    def clustering(self, n_clusters, data):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=76)
        kmeans.fit(data)
        self.inertia = kmeans.inertia_
        self.silhouette_score = metrics.silhouette_score(data, kmeans.labels_, metric='euclidean', sample_size=10000)
        self.centers = kmeans.cluster_centers_
        self.clustered_labels = kmeans.labels_
        self.clustered_img = self.clustered_labels.reshape(647, 993)
        return self.clustered_img

    def get_mapping(self):
        sorted_indices = sorted(range(len(self.centers)), key=lambda i: sum(self.centers[i]))
        cluster_mapping = {original_index: sorted_index for sorted_index, original_index in enumerate(sorted_indices)}
        return cluster_mapping

    def apply_mapping_to_labels(self, mapping):
        self.clustered_img = np.vectorize(mapping.get)(self.clustered_img)
        return self.clustered_img

    def get_silhouette_score(self):
        return self.silhouette_score

    def get_inertia(self):
        return self.inertia

    def get_centers(self):
        return self.centers
