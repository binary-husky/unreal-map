# Created by yingwen at 2019-03-12
"""
This module creates an OU exploration strategy.
Ornstein Uhlenbeck exploration strategy comes from the Ornstein-Uhlenbeck
process. It is often used in DDPG algorithm because in continuous control task
it is better to have temporally correlated exploration to get smoother
transitions. And OU process is relatively smooth in time.
"""
import numpy as np

from malib.policies.explorations.base_exploration import ExplorationBase
from malib.spaces.utils import flat_dim



class OUExploration(ExplorationBase):
    """
    An OU exploration strategy to add noise to environment actions.
    Args:
        env_spec: Environment for OUStrategy to explore.
        mu: A parameter to simulate the process.
        sigma: A parameter to simulate the process.
        theta: A parameter to simulate the process.
        dt: A parameter to simulate the process.
        x0: Initial state.
    Example:
        $ python garage/tf/exploration_strategies/ou_strategy.py
    """
    def __init__(self, action_space, mu=0, sigma=0.5, theta=0.3, dt=1e-2,
                 x0=None):
        self.action_space = action_space
        self.action_dim = flat_dim(self.action_space)
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def simulate(self):
        """
        Compute the next state of the exploration.
        Returns:
            self.state: Next state of the exploration.
        """
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=len(x))
        self.state = x + dx
        return self.state

    def reset(self):
        """Reset the state of the exploration."""
        self.state = self.x0 if self.x0 is not None else self.mu * np.zeros(
            self.action_dim)

    def get_action(self, t, observation, policy, extend_dim=False, **kwargs):
        """Return an action with noise.
        Args:
            t: Iteration.
            observation: Observation from the environment.
            policy: Policy network to predict action based on the observation.
        Returns:
            An action with noise explored by OUStrategy.
        """

        action = policy.get_action_np(observation, extend_dim=extend_dim)
        ou_state = self.simulate()
        return np.clip(action + ou_state, self.action_space.low,
                       self.action_space.high)

    def get_actions(self, t, observations, policy, **kwargs):
        actions = policy.get_actions(observations)
        ou_state = self.simulate()
        return np.clip(actions + ou_state, self.action_space.low,
                       self.action_space.high)


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    env = gym.make("Pendulum-v0")

    ou = OUExploration(
        env.action_space, mu=0, theta=0.15, sigma=0.3)

    states = []
    for i in range(1000):
        states.append(ou.simulate()[0])

    plt.plot(states)
    plt.show()