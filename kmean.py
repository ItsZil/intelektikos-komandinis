import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from plotter import Plotter

class KMeans:
    def __init__(self, n_clusters=3, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None

    def fit(self, X):
        # Randomly initialize centroids
        random_indices = np.random.choice(range(len(X)), size=self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iterations):
            # Assign data points to the nearest centroid
            labels = self._assign_labels(X)

            # Update centroids
            new_centroids = self._update_centroids(X, labels)

            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def _assign_labels(self, X):
        labels = []
        for point in X:
            distances = np.linalg.norm(point - self.centroids, axis=1)
            label = np.argmin(distances)
            labels.append(label)
        return np.array(labels)

    def _update_centroids(self, X, labels):
        new_centroids = []
        for cluster in range(self.n_clusters):
            cluster_points = X[labels == cluster]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
            else:
                centroid = self.centroids[cluster]  # Keep the same centroid if no points in cluster
            new_centroids.append(centroid)
        return np.array(new_centroids)


def getDataFile(fileName):
    data = pd.read_csv(fileName)

    data = data.dropna()
    data = data.drop(data.columns[0], axis=1)

    # Drop unnecessary columns
    columns_to_drop = ['trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'unix_time', 'is_fraud']
    data = data.drop(columns_to_drop, axis=1)

    # Convert data types
    data = pd.get_dummies(data, columns=['gender', 'category'])

    # Perform frequency encoding for 'merchant' and 'job' columns
    merchant_counts = data['merchant'].value_counts()
    data['merchant'] = data['merchant'].map(merchant_counts)

    job_counts = data['job'].value_counts()
    data['job'] = data['job'].map(job_counts)

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    return data

def main():
    data = getDataFile('Data/fraudTrain.csv')
    plotter = Plotter(data)

    #kmeans = KMeans(n_clusters=5)
    #kmeans.fit(data.values)
    #cluster_labels = kmeans._assign_labels(data.values)

    print(data)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(data)
    cluster_labels = kmeans.labels_

    print(cluster_labels)


if __name__ == '__main__':
    main()