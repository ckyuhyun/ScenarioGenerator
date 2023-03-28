import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
This class draws circles representing a node and a classification where has nodes.
'''
class ClassificationGraphy:
    def __init__(self, data: pd.DataFrame, x_label, y_label):
        self.axis_width = 100
        self.axis_height = 100
        self._data = data
        self._x_label = x_label
        self._y_label = y_label
        self.coordinates = {}

        self.generate_coordinate()

    def Draw(self):
        x_coordinate = [x for x, y in list(self.coordinates.values())]
        y_coordinate = [y for x, y in list(self.coordinates.values())]

        plt.scatter(x_coordinate, y_coordinate,  alpha=0.3)
        plt.text(x_coordinate[0], y_coordinate[0],"Base Action")

        center_x, center_y = x_coordinate[0], y_coordinate[0]
        for x, y in zip(x_coordinate[1:], y_coordinate[1:]):
            plt.plot(np.array([center_x, x]), np.array([center_y, y]))


        plt.xlabel(self._x_label)
        plt.ylabel(self._y_label)
        plt.axis('off')
        plt.show()

    '''
    Generate coordinates for each action 
    '''
    def generate_coordinate(self):
        _data = np.concatenate((self._data[self._data.columns[0]].values,self._data[self._data.columns[1]].values))
        for _d in _data:
            if _d not in self.coordinates.keys():
                coordinates = np.array([np.random.randint(0,self.axis_width), np.random.randint(0,self.axis_height)])
                self.coordinates[_d] = coordinates







