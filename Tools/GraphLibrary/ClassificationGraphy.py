import random

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
        self.base_node = {}
        self.init()

    def init(self):
        self.generate_relation_coordinate()
        # self.base_node = self.coordinates.keys()[0]

    def Run(self):
        self.generate_relation_coordinate()
        keys = []
        for _key in self.coordinates.keys():
            keys.append(_key)

        # first start
        _key = random.choices(keys)[0]
        near_nodes = self._data.loc[self._data[self._data.columns[0]] == _key, self._data.columns[1]]
        coordinate = [self.coordinates[node_key] for node_key in near_nodes.values]

        try:
            center_x, center_y = coordinate[0]
        except:
            _key = random.choices(keys)[0]
            near_nodes = self._data.loc[self._data[self._data.columns[0]] == _key, self._data.columns[1]]
            coordinate = [self.coordinates[node_key] for node_key in near_nodes.values]

            center_x, center_y = coordinate[0]


        for i in range(0, 10):
            self.draw(center_x,
                      center_y,
                      np.array(coordinate)[1:, :1],
                      np.array(coordinate)[1:, 1:2],
                      self._x_label,
                      self._y_label)

            _key = random.choice(list(near_nodes.values))
            coordinate = [self.coordinates[node_key] for node_key in near_nodes.values]
            random_index = random.randint(0, len(near_nodes.values) - 1)

            try:
                center_x, center_y = coordinate[random_index]
            except IndexError as e:
                print("Error - {0} : {1} - {2}".format(str(e), random_index, len(coordinate)))


    def draw(self, center_x, center_y, x_coorindates, y_coordinates, x_label, y_label):
        plt.clf()
        plt.scatter(x_coorindates, y_coordinates, alpha=0.3)
        plt.text(center_x, center_y, "Base Action")

        for x, y in zip(x_coorindates, y_coordinates):
            plt.plot(np.array([center_x, x[0]]), np.array([center_y, y[0]]))

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.axis('off')

        plt.show(block=False)
        plt.pause(2)





    '''
    Generate coordinates for each action 
    '''

    def generate_relation_coordinate(self):
        _data = np.concatenate((self._data[self._data.columns[0]].values, self._data[self._data.columns[1]].values))
        for _d in _data:
            if _d not in self.coordinates.keys():
                coordinates = np.array([np.random.randint(0, self.axis_width), np.random.randint(0, self.axis_height)])
                self.coordinates[_d] = coordinates
