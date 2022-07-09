import random

import tensorflow as tf

from malib.trainers.utils import *
from malib.utils import tf_utils


class AdHocSeqTrainer:
    def __init__(self, env, agent, teammates, sampler, batch_size, exploration_epochs, adhoc_epochs,
                 teammate_batch_size, recent_batch_size, extra_experiences, save_path=None):
        self.env = env
        self.agent = agent
        self.teammates = teammates
        self.sampler = sampler

        self.batch_size = batch_size
        self.teammate_batch_size = teammate_batch_size
        self.recent_batch_size = recent_batch_size

        self.extra_experiences = extra_experiences
        self.save_path = save_path

        self.exploration_epochs = exploration_epochs
        self.adhoc_epochs = adhoc_epochs

        self.step = 0
        self.episode_step = 0
        self.losses = []
        self.agents = [self.agent, None]

    def run(self):
        for teammate in self.teammates:
            self._setup_teammate(teammate)
            for _ in range(self.exploration_epochs):
                self.agent.use_descended_replay_buffer = None
                self.agent.use_descended_meta_prior = False
                while self.sampler.sample(explore=True):
                    pass

        teammate_batch = np.random.choice(self.teammates, self.teammate_batch_size)
        # random.sample(, )

        for teammate in teammate_batch:
            self._setup_teammate(teammate)
            print('teammate setup')
            self.step = 0
            for epoch in range(self.adhoc_epochs):
                self.episode_step = 0
                self.sampler.sample()
                self.step += 1
                batches = self.sample_batches()
                for extra_experience in self.extra_experiences:
                    if extra_experience == 'annealing':
                        batches = add_annealing(batches, self.step, annealing_scale=1.)
                    # print('annealing', batches[0]['annealing'])
                    elif extra_experience == 'target_actions':
                        batches = add_target_actions(batches, self.agents, self.batch_size)
                    elif extra_experience == 'recent_experiences':
                        batches = add_recent_batches(batches, self.agents, self.recent_batch_size)
                    # if self._enough_recent_experience():
                    # 	batches = [add_recent_batches([batches[0]], [self.agents[0]],
                    # 								  min(self.recent_batch_size, self.episode_step))[0],
                    # 			   add_recent_batches([batches[1]], [self.agents[1]], self.batch_size)[0]]
                    # else:
                    # 	batches = [batches[0],
                    # 			   add_recent_batches([batches[1]], [self.agents[1]], self.batch_size)[0]]
                agents_losses = []
                if self.step % 1 == 0:
                    for i, (agent, batch) in enumerate(zip(self.agents, batches)):
                        agent_losses = agent.train(batch)
                        agents_losses.append(agent_losses)
                    self.losses.append(agents_losses)
                    print('agent 1', self.losses[-1][0])
                    print('agent 2', self.losses[-1][1])

    def sample_batches(self):
        batches = []
        indices = self.agent.replay_buffer.random_indices(self.batch_size)
        for agent in self.agents:
            batch = agent.replay_buffer.batch_by_indices(indices)
            batches.append(batch)
        for extra_experience in self.extra_experiences:
            if extra_experience == 'annealing':
                batches = add_annealing(batches, self.step * 5, annealing_scale=.1)
            elif extra_experience == 'target_actions':
                batches = add_target_actions(batches, self.agents, self.batch_size)
            elif extra_experience == 'recent_experiences':
                batches = add_recent_batches(batches, self.agents, self.batch_size)
        return batches

    def _setup_teammate(self, teammate):
        self.agents = [self.agent, teammate]
        self.sampler.initialize(self.env, self.agents)

    def _enough_recent_experience(self):
        recent_batch_size = min(self.recent_batch_size, self.episode_step)
        return recent_batch_size >= int(self.recent_batch_size / 2)
