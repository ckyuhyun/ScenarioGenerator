from sklearn.model_selection import train_test_split
from tensorflow import keras


class KNN_model:
    def __init__(self):
        self.seed_data = None
        self.__encoding_dim = 32

    def get_seed_data(self, data):
        self.seed_data = data

    def Run(self):
        train_data, test_data = train_test_split(self.seed_data, train_size=0.7, random_state=42)

        input_tensor = keras.Input(shape=train_data.shape[1],)

        encoded = keras.layers.Dense(units=self.__encoding_dim, activation='relu')(input_tensor)
        decoded = keras.layers.Dense(units=train_data.shape[1], activation='sigmoid')(encoded)

        autoencoder = keras.Model(input_tensor, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(train_data, train_data, epochs=10, batch_size=16, verbose=False)
        predicted_matrix = autoencoder.predict(test_data)
        print('predict value : {0}'.format(predicted_matrix))




