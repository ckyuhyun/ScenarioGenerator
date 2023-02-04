import random

import numpy as np
from typing import Any

from tf_agents.environments import py_environment, TFPyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from tf_agents.specs import array_spec, tensor_spec


class AutoScenarioEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, minimum=(0, 0, 0),
                                                        maximum=(1, 1, 1))
        self._observation_spec = array_spec.BoundedArraySpec(shape=(5,), dtype=np.float32, minimum=0)
        self._state = 0

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        _random_value = random.randint(0, 9)
        if _random_value > 7:
            return ts.termination(self.get_observation(),  reward=self.get_rewards())
        else:
            return ts.transition(self.get_observation(), reward=0, discount=0.0)

    def _reset(self) -> ts.TimeStep:
        return ts.restart(np.array([0,0,0,0,0], dtype=np.float32))

    @staticmethod
    def get_observation():
        return np.array([random.randint(0,1), random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)], dtype=np.float32)

    @staticmethod
    def get_rewards():
        return random.randint(0,1)


