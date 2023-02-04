from AutoScenarioEnv import AutoScenarioEnv
from tf_agents.environments import utils

env = AutoScenarioEnv()
utils.validate_py_environment(env, episodes=5)
print(env.time_step_spec())
print(env.observation_spec())
