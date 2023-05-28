import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataProvider import DataProvider
from minisom import MiniSom
from sklearn.metrics import silhouette_score, davies_bouldin_score

def scatterPlot(cluster_index, dataProvider, som):
    for c in np.unique(cluster_index):
        plt.scatter(dataProvider.data.values[cluster_index == c, 0],
                dataProvider.data.values[cluster_index == c, 1], label='cluster='+str(c+1), alpha=.7)

    # plotting centroids
    for centroid in som.get_weights():
        plt.scatter(centroid[:, 0], centroid[:, 1], marker='o',
                    s=150, color='r', label='Klasterių centrai')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Self-Organizing Map - Scatter grafikas')
    plt.legend()
    plt.show()

def find_optimal_cluster_count(data):
    silhouette_scores = []
    cluster_range = range(2, 11)
    sample_data = data.sample(n=50000)

    for k in cluster_range:
        som = MiniSom(k, 1, data.shape[1], sigma=0.3, learning_rate=0.5)
        som.random_weights_init(sample_data.values)
        som.train_batch(sample_data.values, 100)

        bmu_indexes = som.win_map(sample_data.values)
        sample_labels = np.zeros(len(sample_data))
        for i, indexes in enumerate(bmu_indexes.values()):
            sample_labels[np.concatenate([index for index_list in indexes for index in index_list])] = i

        silhouette_coefficient = silhouette_score(sample_data.values, sample_labels, metric='euclidean')
        silhouette_scores.append(silhouette_coefficient)

    plt.plot(cluster_range, silhouette_scores, 'bx-')
    plt.title('Optimalių klasterių skaičiaus radimas: Silhouette metodas')
    plt.xlabel('Klasterių skaičius (k)')
    plt.ylabel('Silhouette koeficientas')
    plt.show()

def evaluate_errors(data, som, cluster_index):
    quantization_error = np.mean(np.linalg.norm(data - som.quantization(data), axis=1))
    silhouette_coef = silhouette_score(data, cluster_index)
    topographic_error = som.topographic_error(data)
    davies_bouldin_index = davies_bouldin_score(data, cluster_index)
        
    print(f"Quantization Error: {quantization_error}")
    print(f"Topographic Error: {topographic_error}")
    print(f"Sillhouette coefficient: {silhouette_coef}")
    print(f'Davies-Bouldin index: {davies_bouldin_index}')

def main():
    dataProvider = DataProvider('Data/smallData.csv')
    
    columns_to_keep = ['lat', 'long']
    #columns_to_keep = ['amt', 'city_pop']
    
    #ground_truth_labels = dataProvider.data['is_fraud'].values
    dataProvider.processData(columns_to_keep)

    #dataProvider.listInfo()
    #dataProvider.plotData()
    
    som_shape = (1, 4)
    som = MiniSom(1, 4, dataProvider.data.shape[1], sigma=0.3, learning_rate=0.5)
    som.random_weights_init(dataProvider.data.values)
    som.train_batch(dataProvider.data.values, 10000)

    winner_coordinates = np.array([som.winner(x) for x in dataProvider.data.values]).T
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    
    evaluate_errors(dataProvider.data.values, som, cluster_index)

    # Plot results
    scatterPlot(cluster_index, dataProvider, som)


if __name__ == '__main__':
    main()
