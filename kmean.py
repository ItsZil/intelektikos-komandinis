import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataProvider import DataProvider
from sklearn.metrics import silhouette_score, davies_bouldin_score

class KMeans:
    def __init__(self, n_clusters=4, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None

    def fit(self, X):
        random_indices = np.random.choice(range(len(X)), size=self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iterations):
            labels = self.assign_labels(X)
            new_centroids = self.update_centroids(X, labels)

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def assign_labels(self, X):
        labels = []
        for point in X:
            distances = np.linalg.norm(point - self.centroids, axis=1)
            label = np.argmin(distances)
            labels.append(label)
        return np.array(labels)

    def update_centroids(self, X, labels):
        new_centroids = []
        for cluster in range(self.n_clusters):
            cluster_points = X[labels == cluster]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
            else:
                centroid = self.centroids[cluster]
            new_centroids.append(centroid)
        return np.array(new_centroids)
    
    def calculate_inertia(self, labels, data):
        inertia = 0
        for i in range(len(data)):
            centroid = self.centroids[labels[i]]
            inertia += np.linalg.norm(data[i] - centroid) ** 2
        return inertia



def scatterPlot1(data, cluster_labels, centroids):
    plt.scatter(data['amt'], data['city_pop'], c=cluster_labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='red', s=150, label='Klasterių centrai')
    plt.xlabel('Suma')
    plt.ylabel('Miesto gyventojų skaičius')
    plt.title('K-vidurkių klasteriai - Scatter grafikas')

    plt.legend()
    plt.show()

def scatterPlot2(data, cluster_labels, centroids):
    plt.scatter(data['lat'], data['long'], c=cluster_labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='red', s=150, label='Klasterių centrai')
    plt.xlabel('Platuma')
    plt.ylabel('Ilguma')
    plt.title('K-vidurkių klasteriai - Scatter grafikas')

    plt.legend()
    plt.show()

def find_optimal_cluster_count(data):
    silhouette_coefficients = []
    cluster_range = range(2, 11)
    sample_data = data.sample(n=50000)

    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(sample_data.values)
        inertia.append(kmeans.calculate_inertia(kmeans.assign_labels(sample_data.values), sample_data.values))

    plt.plot(range(1, 11), inertia, 'bx-')
    plt.title('Optimalių klasterių skaičiaus radimas: Elbow metodas')
    plt.xlabel('Klasterių skaičius (k)')
    plt.ylabel('Inercija')
    plt.show()

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(sample_data.values)
        sample_labels = kmeans.assign_labels(sample_data.values)
        silhouette_coefficient = silhouette_score(sample_data.values, sample_labels, n_jobs=16)
        silhouette_coefficients.append(silhouette_coefficient)

    plt.plot(cluster_range, silhouette_coefficients, 'bx-')
    plt.title('Optimalių klasterių skaičiaus radimas: Silhouette metodas')
    plt.xlabel('Klasterių skaičius (k)')
    plt.ylabel('Silhouette koeficientas')
    plt.show()

def get_performance(data, cluster_labels, kmeans):
    silhouette_coefficient = silhouette_score(data.values, cluster_labels, n_jobs=12)

    print(f'Klasterių inercija: {kmeans.calculate_inertia(cluster_labels, data.values)}')
    print(f'Silhouette koeficientas: {silhouette_coefficient}')

def main():
    dataProvider = DataProvider('Data/smallData.csv')

    #columns_to_keep = ['amt', 'city_pop']
    columns_to_keep = ['lat', 'long']
    dataProvider.processData(columns_to_keep)

    #dataProvider.listInfo()
    #dataProvider.plotData()

    find_optimal_cluster_count(dataProvider.data)

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(dataProvider.data.values)

    cluster_labels = kmeans.assign_labels(dataProvider.data.values)
    print(f'Cluster labels: {cluster_labels}')

    get_performance(dataProvider.data, cluster_labels, kmeans)

    # Plot results
    scatterPlot2(dataProvider.data, cluster_labels, kmeans.centroids)

if __name__ == '__main__':
    main()