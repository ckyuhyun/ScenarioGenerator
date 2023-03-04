import random

import numpy as np
from typing import Any

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from tf_agents.specs import array_spec, tensor_spec


class ScenarioGenEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=0,
                                                        maximum=1)
        self._observation_spec = array_spec.BoundedArraySpec(shape=(5,), dtype=np.float32, minimum=0)
        self._state = np.zeros(shape=(5,), dtype=np.float32)

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self):
        return self._state

    def set_state(self, state: Any) -> None:
        pass

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        _random_value = random.randint(0, 9)
        if _random_value > 7:
            return ts.termination(self._state,  reward=1)
        else:
            return ts.transition(self._state, reward=0, discount=0.0)

    def _reset(self) -> ts.TimeStep:
        self._state = np.zeros(shape=(5,), dtype=np.float32)
        return ts.restart(self._state)




