from typing import Any
from .Actions import Actions

import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.specs import array_spec, tensor_spec

import numpy as np
import uuid


class AutoScenarioEnv(py_environment.PyEnvironment):
    def __init__(self):
        """
        Action spec: They could be mostly a next or a previous.
        """
        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.float32,
                                                        minimum=int(Actions.Back),
                                                        maximum=int(Actions.Forward),
                                                        name='TestAction')
        """
        Observation spec
        1. Action Guid
        2. 
        """
        self._observation_spec = array_spec.BoundedArraySpec(shape=(1,),
                                                             dtype=np.float32,
                                                             name="Scenario_Observation")
        """
        State : current state of environment
        """
        self._state = 0
        self._current_reward = 0
        self.is_last_action = False

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass

    def _step(self, action: types.NestedArray):
        if self.is_last_action:
            print('last rewoar {0}'.format(self._current_reward))
            return ts.termination(np.array([self._state], dtype=np.float32), self._current_reward)
        else:
            self._current_reward = self._current_reward + 0.1
            if self._current_reward > 5:
                self.is_last_action = True
            return ts.transition(np.array([self._state], dtype=np.float32), reward=0.0, discount=1.0)

    def _reset(self):
        return ts.restart(np.array([self._state], dtype=np.float32))
