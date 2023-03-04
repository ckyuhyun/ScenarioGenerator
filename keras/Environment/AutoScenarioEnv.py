from typing import Any

import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.specs import array_spec, tensor_spec

import numpy as np

class AutoScenarioEnv(py_environment.PyEnvironment):
    def __init__(self):
        """
        Action spec: They could be mostly a next or a previous.
        """
        self.action_spec = array_spec.BoundedArraySpec(shape=(),
                                                       dtype=np.int32,
                                                       minimum=-1,
                                                       maximum=1,
                                                       name='TestAction')
        """
        Observation spec
        """
        self.observation_spec = array_spec.BoundedArraySpec(shape=(1,),
                                                            dtype=np.float32,
                                                            name="Scenario_Observation")
        """
        State : current state of environment
        """
        self._state = 0

    def observation_spec(self):
        return self.observation_spec

    def action_spec(self):
        return self.action_spec

    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass

    def _step(self, action: types.NestedArray):
        pass

    def _reset(self):
        return ts.restart(np.array([self._state], dtype=np.int32))