import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class KNN_model:
    def __init__(self):
        self.seed_data = None
        self.__encoding_dim = 32
        self.__target_column = 'NextActionGuid_label'
        self.model = KNeighborsClassifier(n_neighbors=5)

    def get_seed_data(self, data: pd.DataFrame):
        # corr = data.corr(method="pearson")
        #
        # mask = np.zeros_like(corr)
        # mask[np.triu_indices_from(mask)] = True
        #
        # # Colors
        # cmap = sns.diverging_palette(240, 10, as_cmap=True)
        # # Plotting the heatmap
        #
        # sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0)
        # plt.show()

        #data_covariance = data.cov()
        #sns.heatmap(data_covariance,vmax=.8, square=True)
        # data_corr = data.corr()
        # sns.heatmap(data_corr, vmax=.8, square=True)
        # plt.show()

        self.seed_data = data
        self.seed_data.to_csv('seed_data.csv')

    def Run(self) -> int:
        corr = self.seed_data.corr(method="pearson")

        # zero_like gives a zero numpy array similar to what is passed as first argument
        # np.triu_indices_from gives the upper triangle indices (read triangle-upper-indices)


        # mask = np.zeros_like(corr)
        # mask[np.triu_indices_from(mask)] = True
        #
        # # Colors.
        # cmap = sns.diverging_palette(240, 10, as_cmap=True)
        # # Plotting the heatmap
        #
        # sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0)
        # plt.show()
        #

        _result = self.seed_data.isna()
        X = np.array(self.seed_data.drop([self.__target_column], axis=1))
        Y = np.array(self.seed_data[self.__target_column])
        #print(self.seed_data[self.__target_column].value_counts())
        x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(X, Y, train_size=0.7, random_state=42, stratify=Y)

        '''
        input_tensor = keras.Input(shape=train_data.shape[1],)

        encoded = keras.layers.Dense(units=self.__encoding_dim, activation='relu')(input_tensor)
        decoded = keras.layers.Dense(units=train_data.shape[1], activation='sigmoid')(encoded)

        autoencoder = keras.Model(input_tensor, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(train_data, train_data, epochs=10, batch_size=16, verbose=False)
        predicted_matrix = autoencoder.predict(test_data)
        print('predict value : {0}'.format(predicted_matrix))
        '''
        score_by_n_neighbors = {}

        # for n_n in range(1, 20):
        #     knn = KNeighborsClassifier(n_neighbors=n_n)
        #
        #     knn.fit(x_train_data, np.ravel(y_train_data))
        #     score = knn.score(x_test_data, np.ravel(y_test_data))
        #     score_by_n_neighbors[n_n] = score
        #
        # for k,v in sorted(score_by_n_neighbors.items(), key=lambda d: d[1], reverse=True):
        #     print(f'n_neighbors : {k} (score : {v})')




        param_grid = {'n_neighbors': np.arange(1, 100)}
        knn_cv = GridSearchCV(self.model, param_grid, cv=5)
        knn_cv.fit(x_train_data, np.ravel(y_train_data))
        print(f'Best Param : {knn_cv.best_params_}')
        print(f'Best score : {knn_cv.best_score_}')


        self.model.fit(x_train_data, np.ravel(y_train_data))
        y_predict = self.model.predict(x_test_data)
        f1 = f1_score(y_test_data, y_predict, average='weighted')
        print(f'Test accuracy :{f1}')
        print('')

    def predict(self, test_data):
        # random_index = random.randint(0, len(x_test_data))
        # reshaped_x_test_data = np.reshape(x_test_data[random_index], (-1, x_test_data[random_index].size))
        y_predict = self.model.predict(test_data)
        #print(f'Predict : {y_predict}')

        # the second index is for a target action
        return y_predict











