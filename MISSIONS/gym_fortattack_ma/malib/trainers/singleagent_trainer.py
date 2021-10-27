"""
The trainer for single-agent training.
"""

import pickle

from malib.trainers.utils import *


class SATrainer:
    """This class implements a single-agent trainer.
    """

    def __init__(self, env, agent, sampler, batch_size=128, steps=10000, exploration_steps=100, training_interval=1,
                 extra_experiences=['target_actions'], save_path=None, ):
        self.env = env
        self.agent = agent
        self.sampler = sampler
        self.batch_size = batch_size
        self.steps = steps
        self.exploration_steps = exploration_steps
        self.training_interval = training_interval
        self.extra_experiences = extra_experiences
        self.losses = []
        self.save_path = save_path

    def setup(self, env, agent, sampler):
        self.env = env
        self.agent = agent
        self.sampler = sampler

    def sample_batches(self):
        indices = self.agent.replay_buffer.random_indices(self.batch_size)
        batch = self.agent.replay_buffer.batch_by_indices(indices)
        return batch

    def individual_forward(self):
        pass

    def apply_gradient(self):
        pass

    def run(self):
        for step in range(self.steps):
            if step < self.exploration_steps:
                self.sampler.sample(explore=True)
                continue
            self.sampler.sample()
            batch = self.sample_batches()
            for extra_experience in self.extra_experiences:
                # NOTE : add_annealing is written for MAgent setting.
                if extra_experience == 'annealing':
                    batch = add_annealing([batch], step - self.exploration_steps, annealing_scale=1.)[0]
                # print('annealing', batch['annealing'])
                elif extra_experience == 'target_actions':
                    batch = add_target_actions([batch], step - self.exploration_steps, self.batch_size)[0]
                elif extra_experience == 'recent_experiences':
                    batch = add_recent_batches([batch], step - self.exploration_steps, 128)[0]

            if step % self.training_interval == 0:
                loss = self.agent.train(batch)
                self.losses.append(loss)

    def save(self):
        if self.save_path is None:
            self.save_path = '/tmp/agents.pickle'
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.agents, f, pickle.HIGHEST_PROTOCOL)

    def restore(self, restore_path):
        with open(restore_path, 'rb') as f:
            self.agents = pickle.load(f)

    def resume(self):
        pass

    def log_diagnostics(self):
        pass
