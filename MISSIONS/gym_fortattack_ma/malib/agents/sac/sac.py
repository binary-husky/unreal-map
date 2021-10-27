# Created by yingwen at 2019-03-16

import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tf.enable_eager_execution()
from malib.agents.base_agent import OffPolicyAgent
from malib.core import Serializable
from malib.utils import tf_utils



class SACAgent(OffPolicyAgent):
    def __init__(self,
                 env_specs,  # (obs_space, act_space)
                 policy,
                 qfs,
                 vf,
                 replay_buffer,
                 policy_optimizer=tf.optimizers.Adam(),
                 qfs_optimizers=(tf.optimizers.Adam(), tf.optimizers.Adam()),
                 vf_optimizer=tf.optimizers.Adam(),
                 exploration_strategy=None,
                 exploration_interval=10,
                 target_update_tau=0.01,
                 target_update_period=10,
                 td_errors_loss_fn=None,  # TODO : fancy to me
                 alpha=0.05,
                 gamma=0.95,
                 reward_scale=1.0,
                 gradient_clipping=None,
                 train_sequence_length=None,
                 name='SAC',
                 agent_id=-1):

        self._Serializable__initialize(locals())
        self._env_specs = env_specs

        observation_space = self._env_specs.observation_space
        action_space = self._env_specs.action_space
        # observation_space = self._env_specs[0]
        # action_space = self._env_specs[1]

        self._exploration_strategy = exploration_strategy

        self._target_vf = Serializable.clone(vf, name='target_vf')

        self._policy_optimizer = policy_optimizer
        self._qfs_optimizers = qfs_optimizers
        self._vf_optimizer = vf_optimizer

        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period

        self._td_errors_loss_fn = (td_errors_loss_fn or tf.losses.Huber)

        self._gamma = gamma
        self._reward_scale = reward_scale
        self._gradient_clipping = gradient_clipping
        self._train_step = 0
        self._exploration_interval = exploration_interval
        self._exploration_status = False

        self.required_experiences = ['observation', 'actions', 'rewards', 'next_observations',
                                     'terminals', 'annealing']

        self._qfs = qfs  # here we have duel q-functions, do it in SAC instead of OffPolicyAgent
        self._vf = vf

        # self.saver = tfe.Saver()
        super(SACAgent, self).__init__(
            observation_space,
            action_space,
            policy,
            qfs,
            replay_buffer,
            train_sequence_length=train_sequence_length,
            name=name,
        )

    def act(self, observation, step=None, use_target=False):
        if use_target is False:
            observation = observation[None]
        if self._exploration_strategy is not None and self._exploration_status:
            if step is None:
                step = self._train_step
            if step % self._exploration_interval == 0:
                self._exploration_strategy.reset()
            return self._exploration_strategy.get_actions(self._train_step, observation, self._policy)
        policy = self._policyz
        # return policy(observation[np.newaxis, :])[0]
        return policy.get_actions_np(observation)

    def init_opt(self):
        tf_utils.soft_variables_update(self._vf.trainable_variables, self._target_vf.trainable_variables, tau=1.0)
        self._exploration_status = True

    def init_eval(self):
        self._exploration_status = False

    def _update_target(self):
        tf_utils.soft_variables_update(self._vf.trainable_variables, self._target_vf.trainable_variables,
                                       tau=self._target_update_tau)

    def _q_train(self, batch, weights=None):
        for qf, qf_optimizer in zip(self._qfs, self._qfs_optimizers):
            qf_variables = qf.trainable_variables
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                assert qf_variables, 'No qf variables to optimize.'
                tape.watch(qf_variables)
                qf_loss = self.q_function_loss(qf, batch['observations'], batch['actions'], batch['rewards'],
                                               batch['next_observations'], batch['terminals'], weights=weights)
            tf.debugging.check_numerics(qf_loss, 'qf loss is inf or nan.')
            qf_grads = tape.gradient(qf_loss, qf_variables)
            tf_utils.apply_gradients(qf_grads, qf_variables, qf_optimizer, self._gradient_clipping)
        return qf_loss

    def _v_train(self, batch, weights=None):
        vf_variables = self._vf.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert vf_variables, 'No vf variables to optimize.'
            tape.watch(vf_variables)
            vf_loss = self.value_function_loss(batch['observations'], batch['annealing'], weights=weights)
        tf.debugging.check_numerics(vf_loss, 'Actor loss is inf or nan.')
        vf_grads = tape.gradient(vf_loss, vf_variables)
        tf_utils.apply_gradients(vf_grads, vf_variables, self._vf_optimizer, self._gradient_clipping)
        return vf_loss

    def _p_train(self, batch, weights=None):
        actor_variables = self._policy.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert actor_variables, 'No actor variables to optimize.'
            tape.watch(actor_variables)
            actor_loss = self.policy_loss(batch['observations'], batch['annealing'], weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, actor_variables)
        tf_utils.apply_gradients(actor_grads, actor_variables, self._policy_optimizer, self._gradient_clipping)
        return actor_loss

    def _train(self, batch, weights=None):
        q_loss = self._q_train(batch)
        v_loss = self._v_train(batch)
        p_loss = self._p_train(batch)
        self._train_step += 1
        if self._train_step % self._target_update_period == 0:
            self._update_target()

        losses = {'q loss': q_loss.numpy(),
                  'value loss': v_loss.numpy(),
                  'policy loss': p_loss.numpy()}
        return losses

    @tf.function
    def policy_loss(self, observations, annealing, weights=None):
        # FIXME : figure out what does weight mean ?
        # actions, log_pis = self._policy(observations)
        actions = self._policy.get_actions([observations])
        log_pis = self._policy.log_pis([observations], actions)

        q_values = tf.squeeze(self._qfs[0].get_values([observations, actions]))
        policy_loss = tf.reduce_mean(annealing * log_pis - q_values)
        return policy_loss

    @tf.function
    def value_function_loss(self, observations, annealing, weights=None):
        values = tf.squeeze(self._vf.get_values([observations]))
        actions = self._policy.get_actions([observations])
        log_pis = self._policy.log_pis([observations], actions)
        q_values = tf.squeeze(self._qfs[0].get_values([observations, actions]))
        td_target = q_values - annealing * log_pis
        value_function_loss = tf.reduce_mean(tf.losses.mean_squared_error(td_target, values))
        return value_function_loss

    @tf.function
    def q_function_loss(self, q_function, observations, actions, rewards, next_observations, terminals, weights=None):
        terminals = tf.cast(terminals, tf.float32)
        next_target_values = tf.squeeze(self._target_vf.get_values([next_observations]))
        q_values = tf.squeeze(q_function.get_values([observations, actions]))
        td_target = tf.stop_gradient(self._reward_scale * rewards + (1 - terminals) * self._gamma * next_target_values)
        q_function_loss = tf.reduce_mean(tf.losses.mean_squared_error(td_target, q_values))
        return q_function_loss

    def save_agent(self, file):   ##
        # save_path = self.saver.save(self._sess, file)
        tf.saved_model.save(self, file)
        print("Model saved in path: %s" % save_path)

    def load_agent(self, file):    ##
        self.saver.restore(self._sess, file)
        print("Loaded agent from", file)