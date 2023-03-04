from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from ScenarioGenEnv import ScenarioGenEnv
from QNetwork import QNetwork
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.environments import utils
from tf_agents.policies import q_policy, random_tf_policy
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.metrics import tf_metrics
import tensorflow as tf

import tensorflow as tf



class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


env = ScenarioGenEnv()
#utils.validate_py_environment(env)
tf_env = tf_py_environment.TFPyEnvironment(env)


fc_layer_params = [32,64,128]
q_net = q_network.QNetwork(
    input_tensor_spec= tf_env.observation_spec(),
    action_spec=tf_env.action_spec(),
    fc_layer_params=fc_layer_params)

_optimizer = tf.keras.optimizers.Adam()
agent = dqn_agent.DqnAgent(time_step_spec=tf_env.time_step_spec(),
                           action_spec=tf_env.action_spec(),
                           q_network=q_net,
                           optimizer=_optimizer)

agent.initialize()

#Replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=1000
)

replay_buffer_observer = replay_buffer.add_batch

initial_collect_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
train_metrics = [tf_metrics.AverageReturnMetric(), tf_metrics.AverageEpisodeLengthMetric()]

init_driver = dynamic_step_driver.DynamicStepDriver(
    env=tf_env,
    policy=initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(100)],
    num_steps=20)

final_time_step, final_policy_state = init_driver.run()
print(final_time_step)
print(final_policy_state)


dataset = replay_buffer.as_dataset(sample_batch_size=64, num_steps=2, num_parallel_calls=3).prefetch(3)




all_train_loss = []
all_metrics = []

def train_agent(n_iterations):
    time_step = None 

    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)

    for iteration in range(n_iterations):
        current_metrics = []

        time_step, policy_state = init_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)

        train_loss = agent.train(trajectories)
        all_train_loss.append(train_loss.loss.numpy())

        for i in range(len(train_metrics)):
            current_metrics.append(train_metrics[i].result().numpy())

        all_metrics.append(current_metrics)

        if iteration % 500 == 0:
            print("\nIteration: {}, loss:{:.2f}".format(iteration, train_loss.loss.numpy()))

            for i in range(len(train_metrics)):
                print('{}: {}'.format(train_metrics[i].name, train_metrics[i].result().numpy()))



train_agent(150)