from tf_agents.networks import network
import tensorflow as tf


class AutoScenarioPolicy(network.Network):
    def __init__(self, input_tensor_spec, output_tensor_spec, action_spec):
        super(AutoScenarioPolicy, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name="AutoScenarioPolicy"
        )
        self._output_tensor_spec = output_tensor_spec
        self._action_spec = action_spec
        self._sub_layers = [
            tf.keras.layers.Dense(
                self._action_spec.shape.num_elements(), activation=tf.nn.tanh),
        ]

    def call(self, observation):
        output = observation

        for layer in self._sub_layers:
            output = layer(output)

        




