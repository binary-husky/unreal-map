# Created by yingwen at 2019-06-30
from copy import deepcopy
import numpy as np
import tensorflow as tf


def add_target_actions(batch_n, agents, batch_size):
    target_actions_n = []
    for i, agent in enumerate(agents):
        print(batch_n[i]['next_observations'].shape)
        target_actions_n.append(agent.act(batch_n[i]['next_observations'], use_target=True))

    for i in range(len(agents)):
        target_actions = target_actions_n[i]
        opponent_target_actions = np.reshape(np.delete(deepcopy(target_actions_n), i, 0), (batch_size, -1))
        target_actions = np.concatenate((target_actions, opponent_target_actions), 1)
        assert target_actions.shape[0] == batch_size
        batch_n[i]['target_actions'] = target_actions
    return batch_n


def add_recent_batches(batches, agents, batch_size):
    for batch, agent in zip(batches, agents):
        recent_batch = agent.replay_buffer.recent_batch(batch_size)
        batch['recent_observations'] = recent_batch['observations']
        batch['recent_actions'] = recent_batch['actions']
        if 'opponent_actions' in recent_batch.keys():
            batch['recent_opponent_actions'] = recent_batch['opponent_actions']
    return batches


def add_annealing(batches, step, annealing_scale=1.):
    annealing = .1 + np.exp(-0.1*max(step-10, 0)) * 500
    annealing = annealing_scale * annealing
    for batch in batches:
        batch['annealing'] = tf.constant(annealing, dtype=tf.float32)
    return batches


def get_batches(agents, batch_size):
    assert len(agents) > 1
    batches = []
    indices = agents[0].replay_buffer.random_indices(batch_size)
    for agent in agents:
        batch = agent.replay_buffer.batch_by_indices(indices)
        batches.append(batch)
    return batches


get_extra_experiences = {
    'annealing': add_annealing,
    'recent_experiences': add_recent_batches,
    'target_actions': add_target_actions,
}