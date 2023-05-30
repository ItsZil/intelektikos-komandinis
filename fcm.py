import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from dataProvider import DataProvider

class FuzzyCMeans:
    def __init__(self, n_clusters=3, max_iterations=100, m=2):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.m = m
        self.centroids = None
        self.u = None

    def fit(self, X):
        # Initialize membership matrix
        self.u = np.random.dirichlet(np.ones(self.n_clusters), size=len(X))

        for _ in range(self.max_iterations):
            # Update centroids
            self.centroids = self.update_centroids(X)

            # Update membership matrix
            new_u = self.update_membership(X)

            # Check for convergence
            if np.allclose(self.u, new_u):
                break

            self.u = new_u

    def update_centroids(self, X):
        um = self.u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T

    def update_membership(self, X):
        power = 2 / (self.m - 1)
        temp = np.zeros((len(X), self.n_clusters))

        for i in range(len(X)):
            x = X[i]
            for j in range(self.n_clusters):
                c = self.centroids[j]
                numerator = np.linalg.norm(x - c)
                denominator = 0
                for k in range(self.n_clusters):
                    denominator += (numerator / np.linalg.norm(x - self.centroids[k])) ** power
                temp[i][j] = 1 / denominator

        return temp

    def assign_labels(self, X):
        return np.argmax(self.u, axis=1)

def scatterPlot1(data, cluster_labels, centroids):
    plt.scatter(data['amt'], data['city_pop'], c=cluster_labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='red', s=150, label='Klasterių centrai')
    plt.xlabel('Suma')
    plt.ylabel('Miesto gyventojų skaičius')
    plt.title('Fuzzy C-Means klasteriai - Scatter grafikas')

    plt.legend()
    plt.show()

def scatterPlot2(data, cluster_labels, centroids):
    plt.scatter(data['lat'], data['long'], c=cluster_labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='red', s=150, label='Klasterių centrai')
    plt.xlabel('Latitudė')
    plt.ylabel('Longitudė')
    plt.title('Fuzzy C-Means klasteriai - Scatter grafikas')

    plt.legend()
    plt.show()

def get_performance(data, cluster_labels):
    silhouette_coefficient = silhouette_score(data.values, cluster_labels, n_jobs=12)
    davies_bouldin_index = davies_bouldin_score(data.values, cluster_labels)

    print(f'Silhouette koeficientas: {silhouette_coefficient}')
    print(f'Davies-Bouldin indeksas: {davies_bouldin_index}')

def main():
    dataProvider = DataProvider('Data/smallData.csv')

    #columns_to_keep = ['amt', 'city_pop']
    columns_to_keep = ['lat', 'long']
    dataProvider.processData(columns_to_keep)

    dataProvider.listInfo()
    dataProvider.plotData()

    fcm = FuzzyCMeans(n_clusters=3)
    fcm.fit(dataProvider.data.values)

    cluster_labels = fcm.assign_labels(dataProvider.data.values)
    print(f'Cluster labels: {cluster_labels}')

    get_performance(dataProvider.data, cluster_labels)

    # Plot results
    scatterPlot2(dataProvider.data, cluster_labels, fcm.centroids)

if __name__ == '__main__':
    main()