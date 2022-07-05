import numpy as np
import stable_baselines3


class env_pass:
    # Initialising the shared resources
    def __init__(self):
        self.act = None
        

    # Add an item for the producer
    def produce_item(self, x_item):
        print("Producer adding an item to the list")
        self.x.append(x_item)
        self.ob = 1

    # Consume an item for the consumer
    def consume_item(self):
        print("Consuming from the list")
        consumed_item = self.x[0]
        print("Consumed item: ", consumed_item)
        self.x.remove(consumed_item)
        self.act = 1


class BaselineFoundation():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.handler = [None for _  in range(self.n_thread)]

        self.baseline_alg = stable_baselines3.SAC(policy="MlpPolicy", env='HybridMultiagentPlayground')

        assert n_agent == 1

    def interact_with_env(self, team_intel):
        info = team_intel['Latest-Team-Info']
        done = team_intel['Env-Suffered-Reset']
        step_cnt = team_intel['Current-Obs-Step']
        action_list = []
        for thread in range(self.n_thread):
            pass
        return action_list, None



'''
import gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
'''