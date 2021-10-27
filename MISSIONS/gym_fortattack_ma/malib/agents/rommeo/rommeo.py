# Created by yingwen at 2019-03-16
import tensorflow as tf

from malib.agents.base_agent import OffPolicyAgent
from malib.core import Serializable
from malib.logger import logger, tabular
from malib.utils import tf_utils
import numpy as np


class ROMMEOAgent(OffPolicyAgent):
    def __init__(self, env_specs, policy, qf, replay_buffer, opponent_policy,
                 policy_optimizer=tf.optimizers.Adam(0.01), qf_optimizer=tf.optimizers.Adam(0.01),
                 opponent_policy_optimizer=tf.optimizers.Adam(0.01),
                 opponent_prior_optimizer=tf.optimizers.Adam(0.01), exploration_strategy=None, target_update_tau=0.01,
                 target_update_period=1, td_errors_loss_fn=None, gamma=0.95, reward_scale=1.0, gradient_clipping=None,
                 train_sequence_length=None, name='ROMMEO', agent_id=-1, uniform=False, custom_b=False, bi=1., bj=1.):
        self._Serializable__initialize(locals())
        self._agent_id = agent_id
        self._env_specs = env_specs
        self._uniform = uniform
        self._custom_b = custom_b
        self._bj = tf.constant(bj, dtype=tf.float32)
        self._bi = tf.constant(bi, dtype=tf.float32)

        if self._agent_id >= 0:
            observation_space = self._env_specs.observation_space[self._agent_id]
            action_space = self._env_specs.action_space[self._agent_id]
            opponent_observation_flat_dim = self._env_specs.observation_space.opponent_flat_dim(
                self._agent_id)
            opponent_action_flat_dim = self._env_specs.action_space.opponent_flat_dim(
                self._agent_id)
        else:
            observation_space = self._env_specs.observation_space
            action_space = self._env_specs.action_space

        self._exploration_strategy = exploration_strategy
        self._target_policy = None

        self._opponent_policy = opponent_policy
        self._prior = Serializable.clone(
            self._opponent_policy, name='prior_{}'.format(self._agent_id), repara=False)

        self._target_qf = Serializable.clone(
            qf, name='target_qf_agent_{}'.format(self._agent_id))

        self._opponent_policy_optimizer = opponent_policy_optimizer
        self._opponent_prior_optimizer = opponent_prior_optimizer
        self._actor_optimizer = policy_optimizer
        self._critic_optimizer = qf_optimizer

        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._td_errors_loss_fn = (td_errors_loss_fn or tf.losses.Huber)
        self._gamma = gamma
        self._reward_scale = reward_scale
        self._gradient_clipping = gradient_clipping
        self._train_step = 0
        self._exploration_interval = 10
        self._exploration_status = True

        self.required_experiences = ['observation', 'actions', 'rewards', 'next_observations', 'opponent_actions',
                                     'terminals', 'annealing', 'recent_observations', 'recent_opponent_actions']

        super(ROMMEOAgent, self).__init__(observation_space, action_space, policy, qf, replay_buffer,
                                          train_sequence_length=train_sequence_length, name=name)

    def act(self, observation, step=None, use_target=False):
        if use_target is False:
            observation = observation[None]
        with self._opponent_policy.set_deterministic(True):
            opponent_action = self._opponent_policy.get_actions_np(observation)
            # print('step deterministic opponent_action i'+self._name, opponent_action)
        # # print(opponent_action, 'opponent_action', observation, 'observation')
        policy_input = [observation, opponent_action]
        policy = self._policy
        if not self._exploration_status:
            with policy.set_deterministic(True):
                action = policy.get_actions_np(policy_input)
                # print('step deterministic action'+self._name, action)
        else:
            action = policy.get_actions_np(policy_input)
            # print('step non-deterministic action'+self._name, action)
        # EPS = 1e-6
        # min_eps = np.full(action.shape, 1. - EPS)
        # max_eps = np.full(action.shape, -1. + EPS)
        # action = np.minimum(min_eps, np.maximum(max_eps, action))
        return action

    def init_opt(self):
        tf_utils.soft_variables_update(
            self._qf.trainable_variables, self._target_qf.trainable_variables, tau=1.0)
        self._exploration_status = True

    def init_eval(self):
        self._exploration_status = False

    def _update_target(self):
        tf_utils.soft_variables_update(self._qf.trainable_variables, self._target_qf.trainable_variables,
                                       tau=self._target_update_tau)

    def _prior_train(self, batch):
        prior_variable = self._prior.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert prior_variable, 'No prior variables to optimize.'
            tape.watch(prior_variable)
            prior_loss = self.prior_loss(
                batch['recent_observations'], batch['recent_opponent_actions'])
        tf.debugging.check_numerics(prior_loss, 'prior loss is inf or nan.')
        prior_grads = tape.gradient(prior_loss, prior_variable)
        tf_utils.apply_gradients(
            prior_grads, prior_variable, self._opponent_prior_optimizer, self._gradient_clipping)
        return prior_loss

    def _opponent_train(self, batch):
        opponent_policy_variable = self._opponent_policy.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert opponent_policy_variable, 'No opponent policy variables to optimize.'
            tape.watch(opponent_policy_variable)
            opponent_policy_loss = self.opponent_policy_loss(
                batch['observations'], batch['annealing'])
        tf.debugging.check_numerics(
            opponent_policy_loss, 'opponent policy loss is inf or nan.')
        opponent_policy_grads = tape.gradient(
            opponent_policy_loss, opponent_policy_variable)
        tf_utils.apply_gradients(opponent_policy_grads, opponent_policy_variable, self._opponent_policy_optimizer,
                                 self._gradient_clipping)
        return opponent_policy_loss

    def _critic_train(self, batch, weights=None):
        critic_variables = self._qf.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert critic_variables, 'No qf variables to optimize.'
            tape.watch(critic_variables)
            critic_loss = self.critic_loss(batch['observations'], batch['actions'], batch['opponent_actions'],
                                           batch['rewards'], batch['next_observations'], batch['terminals'],
                                           batch['annealing'], weights=weights)
        tf.debugging.check_numerics(critic_loss, 'qf loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, critic_variables)
        tf_utils.apply_gradients(
            critic_grads, critic_variables, self._critic_optimizer, self._gradient_clipping)
        return critic_loss

    def _actor_train(self, batch, weights=None):
        actor_variables = self._policy.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert actor_variables, 'No actor variables to optimize.'
            tape.watch(actor_variables)
            actor_loss = self.actor_loss(
                batch['observations'], batch['annealing'], weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, actor_variables)
        tf_utils.apply_gradients(
            actor_grads, actor_variables, self._actor_optimizer, self._gradient_clipping)
        return actor_loss

    # @tf.function
    def _train(self, batch, weights=None):
        prior_loss = self._prior_train(batch)
        opponent_policy_loss = self._opponent_train(batch)
        critic_loss = self._critic_train(batch, weights)
        actor_loss = self._actor_train(batch, weights)

        self._train_step += 1
        if self._train_step % self._target_update_period == 0:
            self._update_target()

        losses = {'pg_loss': actor_loss.numpy(), 'critic_loss': critic_loss.numpy(),
                  'opponent_policy_loss': opponent_policy_loss.numpy(), 'prior_loss': prior_loss.numpy()}

        if self._train_step % 1 == 0:
            tabular.record('q loss', critic_loss.numpy().item())
            tabular.record('opponent_policy_loss loss',
                           opponent_policy_loss.numpy().item())
            tabular.record('actor_loss loss', actor_loss.numpy().item())
            tabular.record('bi', batch['annealing'].numpy())
            tabular.record('bj', 1.)
            if prior_loss is not None:
                tabular.record('prior loss', prior_loss.numpy())
            logger.log(tabular)

        if self._train_step % 100 == 0:
            # print('training statistics')
            # print(self._opponent_policy.get_diagnostics(batch['observations']))
            # print(self._prior.get_diagnostics(batch['observations']))
            opponent_actions = self._opponent_policy.get_actions_np(
                batch['observations'])
            # print(self._policy.get_diagnostics([batch['observations'], opponent_actions]))
            actions = self._policy.get_actions_np(
                [batch['observations'], opponent_actions])
            # print(self._qf.get_diagnostics([batch['observations'], actions, opponent_actions]))

        return losses

    @tf.function
    def prior_loss(self, recent_observations, recent_opponent_actions):
        # opponent_actions = self._opponent_policy.get_actions([recent_observations])
        prior_actions = self._prior.get_actions([recent_observations])
        loss = tf.reduce_mean(tf.losses.mean_squared_error(
            recent_opponent_actions, prior_actions))
        # log_pis = self._prior.log_pis([recent_observations], recent_opponent_actions)
        # loss = -tf.reduce_mean(log_pis)

        reg = self._prior.get_model_regularization_loss()
        return loss + reg

    @tf.function
    def opponent_policy_loss(self, observations, annealing):
        opponent_actions = self._opponent_policy.get_actions([observations])
        opponent_actions_log_pis = self._opponent_policy.log_pis(
            [observations], opponent_actions)
        prior_log_pis = self._prior.log_pis([observations], opponent_actions)
        prior_log_pis = tf_utils.remove_inf_nan(prior_log_pis)
        actions = self._policy.get_actions([observations, opponent_actions])
        actions_log_pis = self._policy.log_pis(
            [observations, opponent_actions], actions)
        q_values = self._qf.get_values(
            [observations, actions, opponent_actions])
        if not self._custom_b:
            if self._uniform:
                opponent_policy_loss = opponent_actions_log_pis - \
                    q_values + annealing * actions_log_pis
            else:
                opponent_policy_loss = opponent_actions_log_pis - \
                    q_values + annealing * actions_log_pis - prior_log_pis
        else:
            opponent_policy_loss = self._bj * \
                (opponent_actions_log_pis - prior_log_pis) - \
                q_values + self._bi * actions_log_pis

        # print('opponent_policy_loss', opponent_policy_loss.shape)
        # tf.debugging.check_numerics(opponent_policy_loss, 'opponent_policy_loss is inf or nan.')
        opponent_policy_loss = tf.reduce_mean(opponent_policy_loss)
        # print('opponent_policy_loss', opponent_policy_loss.shape)
        # tf.debugging.check_numerics(opponent_policy_loss, 'opponent_policy_loss is inf or nan.')
        return opponent_policy_loss  # + self._opponent_policy.get_model_regularization_loss()

    @tf.function
    def critic_loss(self, observations, actions, opponent_actions, rewards, next_observations, terminals, annealing,
                    weights=None):
        next_opponent_actions = self._opponent_policy.get_actions(
            [next_observations])
        next_opponent_actions_log_pis = self._opponent_policy.log_pis(
            [next_observations], next_opponent_actions)
        next_prior_log_pis = self._prior.log_pis(
            [next_observations], next_opponent_actions)
        next_prior_log_pis = tf_utils.remove_inf_nan(next_prior_log_pis)

        next_actions = self._policy.get_actions(
            [next_observations, next_opponent_actions])
        next_actions_log_pis = self._policy.log_pis(
            [next_observations, next_opponent_actions], next_actions)

        q_value_targets = self._target_qf.get_values(
            [next_observations, next_actions, next_opponent_actions])
        # print('q_value_targets', q_value_targets.shape)
        if not self._custom_b:
            if self._uniform:
                q_value_targets = q_value_targets - annealing * \
                    next_actions_log_pis - next_opponent_actions_log_pis
            else:
                q_value_targets = q_value_targets - annealing * next_actions_log_pis - \
                    next_opponent_actions_log_pis + next_prior_log_pis
        else:
            q_value_targets = q_value_targets + self._bj * \
                (next_prior_log_pis - next_opponent_actions_log_pis) - \
                self._bi * next_actions_log_pis
        # print('q_value_targets', q_value_targets.shape)
        q_value_targets = tf.squeeze(q_value_targets)
        q_value_targets = tf_utils.remove_inf_nan(q_value_targets)
        td_targets = tf.stop_gradient(
            self._reward_scale * rewards + (1 - terminals) * self._gamma * q_value_targets)
        # print('td_targets', td_targets.shape)
        td_targets = tf.squeeze(td_targets)
        q_values = self._qf.get_values(
            [observations, actions, opponent_actions])
        q_values = tf.squeeze(q_values)
        # print('q_values', q_values.shape)
        # critic_loss = 0.5 * tf.square(td_targets - q_values)
        critic_loss = self._td_errors_loss_fn(reduction=tf.losses.Reduction.NONE)(td_targets, q_values)
        # print('critic_loss', critic_loss.shape)

        if weights is not None:
            critic_loss = weights * critic_loss
        critic_loss = tf.reduce_mean(critic_loss)
        # print('critic_loss', critic_loss.shape, critic_loss)
        return critic_loss

    @tf.function
    def actor_loss(self, observations, annealing, weights=None):
        opponent_actions = self._opponent_policy.get_actions(observations)
        actions = self._policy.get_actions([observations, opponent_actions])

        actions_log_pis = self._policy.log_pis(
            [observations, opponent_actions], actions)

        q_values = self._qf.get_values(
            [observations, actions, opponent_actions])

        if weights is not None:
            q_values = weights * q_values

        q_values = tf.squeeze(q_values)

        if not self._custom_b:
            actor_loss = tf.reduce_mean(annealing * actions_log_pis - q_values)
        else:
            actor_loss = tf.reduce_mean(self._bi * actions_log_pis - q_values)

        return actor_loss  # + self._policy.get_model_regularization_loss()
