import pandas as pd

def getDataFile(fileName):
    data = pd.read_csv(fileName)

    data = data.dropna()
    return data

def main():
    data = getDataFile('Data/fraudTrain.csv')
    print(data)

if __name__ == '__main__':
    main()