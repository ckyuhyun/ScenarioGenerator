import tensorflow as tf


class AutoScenarioModel(tf.keras.Model):
    def __init__(self, num_state, hidden_units, num_actions):
        super(AutoScenarioModel, self).__init__()

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_state,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, activation='tanh', kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(
            units=num_actions,
            activation='linear',
            kernel_initializer='RandomNormal'
        )

    def call(self, inputs, training=None, mask=None):
        z = self.input_layer(inputs)

        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layer(z)
        return output














