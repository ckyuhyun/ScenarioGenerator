from tf_agents.networks import network
import tensorflow as tf

import numpy as np


# This policy is along with Q-Network since a primary expectation with this policy
# will generate all of possible actions based on the current time_step(action)
class QNetwork(network.Network):
    def __init__(self, input_tensor_spec, name=None):
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            name=name
        )
        self._init_input_shape = input_tensor_spec.shape.num_elements()
        self._sub_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self._init_input_shape, dtype=np.float32)
        ]

    def call(self, inputs, step_type=None, network_state=()):
        del step_type

        inputs = tf.cast(inputs, tf.float32)
        for layer in self._sub_layers:
            inputs = layer(inputs)

        return inputs, network_state








        




