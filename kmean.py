import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotter import Plotter
#from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

class KMeans:
    def __init__(self, n_clusters=3, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None

    def fit(self, X):
        # random centroid initialization
        random_indices = np.random.choice(range(len(X)), size=self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iterations):
            labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X, labels)

            # konverguoja?
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
                centroid = self.centroids[cluster]
            new_centroids.append(centroid)
        return np.array(new_centroids)
    
    def calculate_inertia(self, labels, data):
        inertia = 0
        for i in range(len(data)):
            centroid = self.centroids[labels[i]]
            inertia += np.linalg.norm(data[i] - centroid) ** 2
        return inertia



def scatterPlot(data, cluster_labels):
    plt.scatter(data['amt'], data['city_pop'], c=cluster_labels, cmap='viridis')
    plt.xlabel('Suma')
    plt.ylabel('Miesto gyventojų skaičius')
    plt.title('K-vidurkių klasteriai - Scatter grafikas')

    unique_labels = np.unique(cluster_labels)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i), markersize=8, label=f'Cluster {label}') for i, label in enumerate(unique_labels)]
    plt.legend(handles=legend_elements)
    plt.show()

def clusterCenters(data, kmeans, cluster_labels):
    plt.scatter(data['amt'], data['city_pop'], c=cluster_labels, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, color='red', label='Klasteriai')
    plt.xlabel('Suma')
    plt.ylabel('Miesto gyventojų skaičius')
    plt.title('K-vidurkių klasteriai')
    plt.legend()

def getDataFile(fileName):
    data = pd.read_csv(fileName)

    # Drop columns
    columns_to_drop = [data.columns[0], 'merchant', 'job', 'lat', 'long', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'unix_time', 'is_fraud', 'merch_lat', 'merch_long']
    data = data.drop(columns_to_drop, axis=1)
    data = data.dropna()

    # Reset index
    data = data.reset_index(drop=True)

    # Convert data types
    data = pd.get_dummies(data, columns=['gender', 'category'])

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    scaler = StandardScaler()
    data[['amt', 'city_pop']] = scaler.fit_transform(data[['amt', 'city_pop']])

    return data

def find_optimal_cluster_count(data):
    silhouette_coefficients = []
    cluster_range = range(2, 11)
    sample_data = data.sample(n=50000)

    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(sample_data.values)
        inertia.append(kmeans.inertia_)
        #inertia.append(kmeans.calculate_inertia(kmeans.labels_, sample_data.values)

    plt.plot(range(1, 11), inertia)
    plt.title('Optimalių klasterių skaičiaus radimas: Elbow metodas')
    plt.xlabel('Klasterių skaičius (k)')
    plt.ylabel('Inertia')
    plt.show()

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(sample_data.values)
        sample_labels = kmeans.labels_
        #sample_labels = kmeans._assign_labels(data.values)
        silhouette_coefficient = silhouette_score(sample_data.values, sample_labels, n_jobs=16)
        silhouette_coefficients.append(silhouette_coefficient)

    plt.plot(cluster_range, silhouette_coefficients, 'bx-')
    plt.title('Optimalių klasterių skaičiaus radimas: Silhouette metodas')
    plt.xlabel('Klasterių skaičius (k)')
    plt.ylabel('Silhouette koeficientas')
    plt.show()

def get_performance(data, cluster_labels, kmeans):
    silhouette_coefficient = silhouette_score(data.values, cluster_labels, n_jobs=12)
    davies_bouldin_index = davies_bouldin_score(data.values, cluster_labels)

    print(f'Cluster inertia: {kmeans.calculate_inertia(cluster_labels, data.values)}')
    print(f'Silhouette Coefficient: {silhouette_coefficient}')
    print(f'Davies-Bouldin Index: {davies_bouldin_index}')

def main():
    data = getDataFile('Data/smallData.csv')
    print(data.columns)
    plotter = Plotter(data)
    
    #find_optimal_cluster_count(data)

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data.values)
    cluster_labels = kmeans._assign_labels(data.values)

    print(data.columns)

    # su scikit:
    #kmeans = KMeans(n_clusters=4, n_init=10)
    #kmeans.fit(data.values)
    #cluster_labels = kmeans.labels_

    print(f'Cluster labels: {cluster_labels}')

    #sample_size = 100000
    #sample_data = data.sample(n=sample_size)
    #sample_labels = kmeans.predict(sample_data.values)

    get_performance(data, cluster_labels, kmeans)

    # Plot results
    scatterPlot(data, cluster_labels)

    plotter.showPlots()


if __name__ == '__main__':
    main()