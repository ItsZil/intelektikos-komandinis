import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class DataProvider:
    def __init__(self, fileName):
        self.data = pd.read_csv(fileName)

    def processData(self, columns_to_keep):
        self.columns = columns_to_keep

        self.data = self.data[columns_to_keep]
        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop=True)

        for column in self.data.columns:
            self.data[column] = pd.to_numeric(self.data[column], errors='coerce')

        scaler = StandardScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

    def listInfo(self):
        print(f'Stulpeliai: {self.data.columns}')
        print(f'Duomenų kiekis: {len(self.data)}')
        print(f'Unikalios reikšmės:\n{self.data.nunique()}')
        print(f'Pirmos 5 eilutės:\n{self.data.head()}')

    def plotData(self):
        plt.scatter(self.data[self.columns[0]], self.data[self.columns[1]], s=1)
        plt.xlabel(self.columns[0])
        plt.ylabel(self.columns[1])
        plt.title('Klasterizavimo duomenys')
        plt.show()