import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier

class KNN_model:
    def __init__(self):
        self.seed_data = None
        self.__encoding_dim = 32

    def get_seed_data(self, data):
        self.seed_data = data

    def Run(self):
        X = self.seed_data[['Action_id_label', 'ActionIdPageLabel', 'ActionName_Label','distance', 'rating']]
        Y = self.seed_data[['Next_id_label']]
        x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(X, Y, train_size=0.7, random_state=42)

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
        '''
        for n_n in range(1, 20):
            knn = KNeighborsClassifier(n_neighbors=n_n)

            knn.fit(x_train_data, np.ravel(y_train_data))
            score = knn.score(x_test_data, np.ravel(y_test_data))
            score_by_n_neighbors[n_n] = score

        for k,v in sorted(score_by_n_neighbors.items(), key=lambda d: d[1], reverse=True):
            print(f'n_neighbors : {k} (score : {v})')
        '''
        knn = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(1, 100)}
        knn_cv = GridSearchCV(knn, param_grid, cv=5)
        knn_cv.fit(x_train_data, np.ravel(y_train_data))
        print(f'Best Param : {knn_cv.best_params_}')
        print(f'Best score : {knn_cv.best_score_}')

        knn = KNeighborsClassifier(n_neighbors=23)
        knn.fit(x_train_data, np.ravel(y_train_data))
        y_predict = knn.predict(x_test_data)
        print(f'Predict : {y_predict}')










