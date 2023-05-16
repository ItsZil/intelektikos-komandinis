import pandas as pd
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, data):
        self.data = data

    def plot(self):
        # manau sitoje klaseje galesim realizuoti grafikus, kuriuos rodysime ataskaitoje aprasant koki duomenu rinkini naudojam
        print(self.data)