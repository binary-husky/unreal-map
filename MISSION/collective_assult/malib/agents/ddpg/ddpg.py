# Created by yingwen at 2019-03-15
import numpy as np
import tensorflow as tf

from malib.agents.base_agent import OffPolicyAgent
from malib.core import Serializable
from malib.utils import tf_utils


class DDPGAgent(OffPolicyAgent):
    def __init__(self, env_specs, policy, qf, replay_buffer, policy_optimizer=tf.optimizers.Adam(),
                 qf_optimizer=tf.optimizers.Adam(), exploration_strategy=None, exploration_interval=10,
                 target_update_tau=0.01, target_update_period=10, td_errors_loss_fn=None, gamma=0.95, reward_scale=1.0,
                 gradient_clipping=None, train_sequence_length=None, name='DDPG', agent_id=-1):
        self._Serializable__initialize(locals())
        self._agent_id = agent_id
        self._env_specs = env_specs
        if self._agent_id >= 0:
            observation_space = self._env_specs.observation_space[self._agent_id]
            action_space = self._env_specs.action_space[self._agent_id]
        else:
            observation_space = self._env_specs.observation_space
            action_space = self._env_specs.action_space

        self._exploration_strategy = exploration_strategy

        self._target_policy = Serializable.clone(policy, name='target_policy_agent_{}'.format(self._agent_id))
        self._target_qf = Serializable.clone(qf, name='target_qf_agent_{}'.format(self._agent_id))

        self._policy_optimizer = policy_optimizer
        self._qf_optimizer = qf_optimizer

        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._td_errors_loss_fn = (td_errors_loss_fn or tf.losses.Huber)
        self._gamma = gamma
        self._reward_scale = reward_scale
        self._gradient_clipping = gradient_clipping
        self._train_step = 0
        self._exploration_interval = exploration_interval
        self._exploration_status = False

        self.required_experiences = ['observation', 'actions', 'rewards', 'next_observations']

        super(DDPGAgent, self).__init__(observation_space, action_space, policy, qf, replay_buffer,
                                        train_sequence_length=train_sequence_length, name=name, )

    def act(self, observation, step=None, use_target=False):
        if use_target:
            # if self._exploration_strategy is not None and self._exploration_status:
            # 	if step is None:
            # 		step = self._train_step
            # 	if step % self._exploration_interval == 0:
            # 		self._exploration_strategy.reset()
            # 	return self._exploration_strategy.get_actions(self._train_step, observation, self._policy)
            # policy = self._policy
            if use_target and self._target_policy is not None:
                policy = self._target_policy
            return policy.get_actions_np(observation)
        else:
            observation = np.array([observation])
            if self._exploration_strategy is not None and self._exploration_status:
                if step is None:
                    step = self._train_step
                if step % self._exploration_interval == 0:
                    self._exploration_strategy.reset()
                return self._exploration_strategy.get_action(self._train_step, observation, self._policy)
            policy = self._policy
            if use_target and self._target_policy is not None:
                policy = self._target_policy
            return policy.get_actions_np(observation)[0]

    def init_opt(self):
        tf_utils.soft_variables_update(self._policy.trainable_variables, self._target_policy.trainable_variables,
                                       tau=1.0)
        tf_utils.soft_variables_update(self._qf.trainable_variables, self._target_qf.trainable_variables, tau=1.0)
        self._exploration_status = True

    def init_eval(self):
        self._exploration_status = False

    def _update_target(self):
        tf_utils.soft_variables_update(self._policy.trainable_variables, self._target_policy.trainable_variables,
                                       tau=self._target_update_tau)
        tf_utils.soft_variables_update(self._qf.trainable_variables, self._target_qf.trainable_variables,
                                       tau=self._target_update_tau)

    def _critic_train(self, batch, weights=None):
        critic_variables = self._qf.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert critic_variables, 'No qf variables to optimize.'
            tape.watch(critic_variables)
            critic_loss = self.critic_loss(batch['observations'], batch['actions'], batch['rewards'],
                                           batch['next_observations'], weights=weights)
        tf.debugging.check_numerics(critic_loss, 'qf loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, critic_variables)
        tf_utils.apply_gradients(critic_grads, critic_variables, self._qf_optimizer, self._gradient_clipping)
        return critic_loss

    def _actor_train(self, batch, weights=None):
        actor_variables = self._policy.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert actor_variables, 'No actor variables to optimize.'
            tape.watch(actor_variables)
            actor_loss = self.actor_loss(batch['observations'], weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, actor_variables)
        tf_utils.apply_gradients(actor_grads, actor_variables, self._policy_optimizer, self._gradient_clipping)
        return actor_loss

    def _train(self, batch, weights=None):
        critic_loss = self._critic_train(batch, weights)
        actor_loss = self._actor_train(batch, weights)

        self._train_step += 1

        if self._train_step % self._target_update_period == 0:
            self._update_target()

        # print('actor_loss + critic_loss', actor_loss , critic_loss)
        total_loss = tf.cast(actor_loss, tf.float32) + tf.cast(critic_loss, tf.float32)

        # print(self._train_step)
        if self._train_step % 1000 == 0:
            print('statistics')
            print(self._policy.get_diagnostics(batch['observations']))
            actions = self._policy.get_actions_np(batch['observations'])
            print(self._qf.get_diagnostics([batch['observations'], actions]))

        return total_loss

    @tf.function
    def critic_loss(self, observations, actions, rewards, next_observations, weights=None):
        """Computes the critic loss for DDPG training.
        Args:
          observations: A batch of observations.
          actions: A batch of actions.
          rewards: A batch of rewards.
          next_observations: A batch of next observations.
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.
        Returns:
          critic_loss: A scalar critic loss.
        """
        target_actions = self._target_policy.get_actions(next_observations)
        target_critic_input = [next_observations, target_actions]
        target_q_values = self._target_qf.get_values(target_critic_input)
        target_q_values = tf.squeeze(target_q_values)
        td_targets = tf.stop_gradient(self._reward_scale * rewards + self._gamma * target_q_values)
        critic_net_input = [observations, actions]
        q_values = self._qf.get_values(critic_net_input)
        q_values = tf.squeeze(q_values)

        critic_loss = self._td_errors_loss_fn(reduction=tf.losses.Reduction.NONE)(td_targets, q_values)

        if weights is not None:
            critic_loss = weights * critic_loss

        critic_loss = tf.reduce_mean(critic_loss)
        return critic_loss

    @tf.function
    def actor_loss(self, observations, weights=None):
        """Computes the actor_loss for DDPG training.
        Args:
          observations: A batch of observations.
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.
          # TODO: Add an action norm regularizer.
        Returns:
          actor_loss: A scalar actor loss.
        """
        actions = self._policy.get_actions(observations)
        q_values = self._qf.get_values([observations, actions])
        q_values = tf.squeeze(q_values)
        if weights is not None:
            q_values = weights * q_values
        actor_loss = -tf.reduce_mean(q_values)
        return actor_loss
