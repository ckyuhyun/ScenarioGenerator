from tf_agents.environments import utils

import Environment


def main():
    env = Environment.AutoScenarioEnv()
    env.reset()
    utils.validate_py_environment(env)



if __name__ == "__main__":
    main()
