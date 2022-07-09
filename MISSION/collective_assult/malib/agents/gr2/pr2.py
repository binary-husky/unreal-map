# Created by yingwen at 2019-03-16

import tensorflow as tf
import numpy as np
from malib.agents.base_agent import OffPolicyAgent
from malib.core import Serializable
from malib.utils import tf_utils
from malib.misc.kernel import adaptive_isotropic_gaussian_kernel
EPS = 1e-6

class PR2Agent(OffPolicyAgent):
    def __init__(self,
                 env_specs,
                 policy,
                 qf,
                 ind_qf,
                 replay_buffer,
                 opponent_policy,
                 policy_optimizer=tf.optimizers.Adam(),
                 qf_optimizer=tf.optimizers.Adam(),
                 opponent_policy_optimizer=tf.optimizers.Adam(),
                 value_n_particles=16,
                 kernel_update_ratio=0.5,
                 exploration_strategy=None,
                 target_update_tau=0.01,
                 target_update_period=1,
                 td_errors_loss_fn=None,
                 gamma=0.95,
                 reward_scale=1.0,
                 gradient_clipping=None,
                 train_sequence_length=None,
                 loss_type='svgd',
                 name='PR2',
                 agent_id=-1
                 ):
        self._Serializable__initialize(locals())
        self._agent_id = agent_id
        self._env_specs = env_specs
        self._value_n_particles = value_n_particles
        self._kernel_update_ratio = kernel_update_ratio
        self._loss_type = loss_type

        observation_space = self._env_specs.observation_space[self._agent_id]
        action_space = self._env_specs.action_space[self._agent_id]
        self._observation_flat_dim = self._env_specs.observation_space.agent_flat_dim(self._agent_id)
        self._action_flat_dim = self._env_specs.action_space.agent_flat_dim(self._agent_id)
        self._opponent_observation_flat_dim = self._env_specs.observation_space.opponent_flat_dim(self._agent_id)
        self._opponent_action_flat_dim = self._env_specs.action_space.opponent_flat_dim(self._agent_id)

        self._exploration_strategy = exploration_strategy
        self._target_policy = None
        self._ind_qf = ind_qf
        self._opponent_policy = opponent_policy
        self._prior = Serializable.clone(self._opponent_policy, name='prior_{}'.format(self._agent_id))

        self._target_policy = Serializable.clone(policy, name='target_policy_agent_{}'.format(self._agent_id))
        self._target_qf = Serializable.clone(qf, name='target_qf_agent_{}'.format(self._agent_id))

        self._actor_optimizer = policy_optimizer
        self._critic_optimizer = qf_optimizer
        self._opponent_policy_optimizer = opponent_policy_optimizer

        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._td_errors_loss_fn = (
                td_errors_loss_fn or tf.losses.Huber)
        self._gamma = gamma
        self._reward_scale = reward_scale
        self._gradient_clipping = gradient_clipping
        self._train_step = 0
        self._exploration_interval = 10
        self._exploration_status = True

        self.required_experiences = ['observation', 'actions', 'rewards', 'next_observations',
                                     'opponent_actions',  'terminals', 'annealing',
                                     'recent_observations', 'recent_opponent_actions']

        super(PR2Agent, self).__init__(
            observation_space,
            action_space,
            policy,
            qf,
            replay_buffer,
            train_sequence_length=train_sequence_length,
            name=name
        )

    def act(self, observation, step=None, use_target=False):
        if self._exploration_strategy is not None and self._exploration_status:
            if step is None:
                step = self._train_step
            if step % self._exploration_interval == 0:
                self._exploration_strategy.reset()
            return self._exploration_strategy.get_action(self._train_step, observation, self._policy)
        policy = self._policy
        if use_target and self._target_policy is not None:
            policy = self._target_policy
        if not self._exploration_status:
            with policy.set_deterministic(True):
                action = policy.get_action_np(observation, extend_dim=True)
            return action
        return policy.get_action_np(observation, extend_dim=True)

    def init_opt(self):
        tf_utils.soft_variables_update(
            self._policy.trainable_variables,
            self._target_policy.trainable_variables,
            tau=1.0)
        tf_utils.soft_variables_update(
            self._qf.trainable_variables,
            self._target_qf.trainable_variables,
            tau=1.0)
        self._exploration_status = True

    def init_eval(self):
        self._exploration_status = False

    def _update_target(self):
        tf_utils.soft_variables_update(
            self._policy.trainable_variables,
            self._target_policy.trainable_variables,
            tau=self._target_update_tau)
        tf_utils.soft_variables_update(
            self._qf.trainable_variables,
            self._target_qf.trainable_variables,
            tau=self._target_update_tau)

    def _train(self, batch, weights=None):

        opponent_policy_variable = self._opponent_policy.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert opponent_policy_variable, 'No opponent policy variables to optimize.'
            tape.watch(opponent_policy_variable)
            opponent_policy_loss = self.opponent_policy_loss(batch['recent_observations'],
                                         batch['recent_actions'],
                                         batch['recent_opponent_actions'])
        tf.debugging.check_numerics(opponent_policy_loss, 'opponent policy loss is inf or nan.')
        opponent_policy_grads = tape.gradient(opponent_policy_loss, opponent_policy_variable)
        tf_utils.apply_gradients(opponent_policy_grads, opponent_policy_variable,
                                 self._opponent_policy_optimizer, self._gradient_clipping)

        critic_variables = self._qf.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert critic_variables, 'No qf variables to optimize.'
            tape.watch(critic_variables)
            critic_loss = self.critic_loss(batch['observations'],
                                           batch['actions'],
                                           batch['opponent_actions'],
                                           batch['rewards'],
                                           batch['next_observations'],
                                           batch['terminals'],
                                           batch['annealing'],
                                           weights=weights)
        tf.debugging.check_numerics(critic_loss, 'qf loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, critic_variables)
        tf_utils.apply_gradients(critic_grads, critic_variables, self._critic_optimizer, self._gradient_clipping)

        actor_variables = self._policy.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert actor_variables, 'No actor variables to optimize.'
            tape.watch(actor_variables)
            actor_loss = self.actor_loss(batch['observations'],
                                         batch['annealing'],
                                         weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, actor_variables)
        tf_utils.apply_gradients(actor_grads, actor_variables, self._actor_optimizer, self._gradient_clipping)
        self._train_step += 1

        if self._train_step % self._target_update_period == 0:
            self._update_target()

        losses = {
            'pg_loss': actor_loss.numpy(),
            'critic_loss': critic_loss.numpy(),
            'opponent_policy_loss': opponent_policy_loss.numpy(),
        }

        return losses

    @tf.function
    def opponent_policy_loss(self,
                              recent_observations,
                              recent_actions,
                              recent_opponent_actions
                            ):
        # log_pis = self._opponent_policy.log_pis([recent_observations, recent_actions], recent_opponent_actions)
        # loss = -tf.reduce_mean(log_pis)

        prior_actions = self._opponent_policy.get_actions([recent_observations, recent_actions])
        loss = tf.reduce_mean(tf.losses.mean_squared_error(
            recent_opponent_actions, prior_actions))
        # log_pis = self._prior.log_pis([recent_observations], recent_opponent_actions)
        # loss = -tf.reduce_mean(log_pis)

        reg = self._prior.get_model_regularization_loss()
        return loss + reg
        # return loss

    @tf.function
    def opponent_svgd_policy_loss(self,
                             observations,
                             actions,
                             annealing
                             ):
        n_updated = int(
            self._value_n_particles * self._kernel_update_ratio)
        n_fixed = self._value_n_particles - n_updated

        @tf.function
        def get_repeated_actions(n):
            repeated_observations = tf.tile(observations[:, None, :], (1, n, 1))
            repeated_observations = tf.reshape(repeated_observations, (-1, self._observation_flat_dim))

            repeated_actions = tf.tile(actions[:, None, :], (1, n, 1))
            repeated_actions = tf.reshape(repeated_actions, (-1, self._action_flat_dim))

            repeated_opponent_actions = self._opponent_policy.get_actions([repeated_observations, repeated_actions])

            repeated_opponent_actions = tf.reshape(repeated_opponent_actions, (-1, n, self._opponent_action_flat_dim))
            return repeated_opponent_actions

        # print('opponent_actions', opponent_actions.shape)
        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.


        # fixed_opponent_actions, updated_opponent_actions = tf.split(
        #     opponent_actions, [n_fixed, n_updated], axis=1)
        fixed_opponent_actions = get_repeated_actions(n_fixed)
        fixed_opponent_actions = tf.stop_gradient(fixed_opponent_actions)

        print('fixed_opponent_actions', fixed_opponent_actions.shape)
        # print('updated_opponent_actions', updated_opponent_actions.shape)
        tf_utils.assert_shape(fixed_opponent_actions, [None, n_fixed, self._opponent_action_flat_dim])

        fixed_observations = tf.reshape(
                    tf.tile(
                        observations[:, None, :], (1, n_fixed, 1)
                    ),
                    (-1, self._observation_flat_dim))

        fixed_actions = tf.reshape(
            tf.tile(
                actions[:, None, :], (1, n_fixed, 1)
            ),
            (-1, self._action_flat_dim)
        )

        @tf.function
        def get_grad_log_p(opponent_actions):
            opponent_actions = tf.reshape(opponent_actions, (-1, self._opponent_action_flat_dim))

            q_values = self._qf.get_values([fixed_observations, fixed_actions, opponent_actions])

            ind_q_values = self._ind_qf.get_values([opponent_actions, fixed_actions])

            q_values = (q_values - ind_q_values) / annealing

            q_values = tf.reshape(q_values, [-1, n_fixed])

            tf_utils.assert_shape(q_values, [None, n_fixed])

            fixed_opponent_actions = tf.reshape(opponent_actions, (-1, n_fixed, self._opponent_action_flat_dim))

            squash_correction = tf.reduce_sum(
                tf.math.log(1 - fixed_opponent_actions ** 2 + EPS), axis=-1)

            log_p = q_values + squash_correction
            return log_p

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(fixed_opponent_actions)
            log_p = get_grad_log_p(fixed_opponent_actions)

        # tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        # actor_grads = tape.gradient(actor_loss, actor_variables)

        grad_log_p = tape.gradient(log_p, fixed_opponent_actions)
        # print('grad_log_p', grad_log_p)
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)
        grad_log_p = tf.stop_gradient(grad_log_p)
        tf_utils.assert_shape(grad_log_p, [None, n_fixed, 1, self._opponent_action_flat_dim])



        # Propagate the gradient through the policy network (Equation 14).

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self._opponent_policy.trainable_variables)
            updated_opponent_actions = get_repeated_actions(n_updated)

        # tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        # actor_grads = tape.gradient(actor_loss, actor_variables)

        kernel_dict = adaptive_isotropic_gaussian_kernel(xs=fixed_opponent_actions, ys=updated_opponent_actions)

        # Kernel function in Equation 13:
        kappa = tf.expand_dims(kernel_dict["output"], axis=3)
        tf_utils.assert_shape(kappa, [None, n_fixed, n_updated, 1])

        # Stein Variational Gradient in Equation 13:
        action_gradients = tf.reduce_mean(
            kappa * grad_log_p + kernel_dict["gradient"], axis=1)
        tf_utils.assert_shape(action_gradients,
                              [None, n_updated, self._opponent_action_flat_dim])

        gradients = tape.gradient(updated_opponent_actions,
                                  self._opponent_policy.trainable_variables,
                                  output_gradients=action_gradients)


        surrogate_loss = tf.reduce_mean([
            tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self._opponent_policy.trainable_variables, gradients)
        ])

        # print()

        return surrogate_loss


    @tf.function
    def critic_loss(self,
                    observations,
                    actions,
                    opponent_actions,
                    rewards,
                    next_observations,
                    terminals,
                    annealing,
                    weights=None):

        next_observations = tf.tile(next_observations[:, None, :], (1, self._value_n_particles, 1))
        next_observations = tf.reshape(next_observations, (-1, self._observation_flat_dim))

        next_actions = self._target_policy.get_actions(next_observations)
        next_opponent_actions = self._opponent_policy.get_actions([next_observations,next_actions])
        q_value_targets = self._target_qf.get_values([next_observations, next_actions, next_opponent_actions])
        q_value_targets = tf.reshape(q_value_targets, (-1, self._value_n_particles))
        # q_value_targets = annealing * tf.reduce_logsumexp(q_value_targets / annealing, keepdims=True, axis=1)
        q_value_targets = tf.reduce_mean(q_value_targets, axis=1)
        q_value_targets = tf.squeeze(q_value_targets)
        td_targets = tf.stop_gradient(self._reward_scale * rewards + (
                1 - terminals) * self._gamma * q_value_targets)
        # td_targets = td_targets - tf.math.log(tf.cast(self._value_n_particles, tf.float32))
        # td_targets = td_targets + self._action_flat_dim * np.log(2)
        td_targets = tf.squeeze(td_targets)
        q_values = self._qf.get_values([observations, actions, opponent_actions])
        q_values = tf.squeeze(q_values)
        ind_q_values = self._ind_qf.get_values([observations, actions])
        ind_q_values = tf.squeeze(ind_q_values)

        critic_loss = self._td_errors_loss_fn(reduction=tf.losses.Reduction.NONE)(td_targets, q_values)
        critic_loss = critic_loss + self._td_errors_loss_fn(reduction=tf.losses.Reduction.NONE)(td_targets, ind_q_values)

        if weights is not None:
            critic_loss = weights * critic_loss

        critic_loss = tf.reduce_mean(critic_loss)

        return critic_loss

    @tf.function
    def actor_loss(self, observations, annealing, weights=None):

        observations = tf.tile(observations[:, tf.newaxis, :], (1, self._value_n_particles, 1))
        observations = tf.reshape(observations, (-1, self._observation_flat_dim))

        actions = self._policy.get_actions(observations)

        opponent_actions = self._opponent_policy.get_actions([observations, actions])

        q_values = self._qf.get_values([observations, actions, opponent_actions])

        q_values = tf.reshape(q_values, (-1, self._value_n_particles))

        # q_values = annealing * tf.reduce_logsumexp(q_values / annealing, axis=1)

        q_values = tf.reduce_mean(q_values, axis=1)

        # q_values = q_values - tf.math.log(tf.cast(self._value_n_particles, tf.float32))
        # q_values = q_values + self._action_flat_dim * np.log(2)

        q_values = tf.squeeze(q_values)

        if weights is not None:
            q_values = weights * q_values

        actor_loss = -tf.reduce_mean(q_values)
        return actor_loss