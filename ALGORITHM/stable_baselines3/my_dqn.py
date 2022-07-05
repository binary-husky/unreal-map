import numpy as np
import stable_baselines3
import torch
from typing import NamedTuple
# class env_pass:
#     # Initialising the shared resources
#     def __init__(self):
#         self.act = None

#     # Add an item for the producer
#     def produce_item(self, x_item):
#         print("Producer adding an item to the list")
#         self.x.append(x_item)
#         self.ob = 1

#     # Consume an item for the consumer
#     def consume_item(self):
#         print("Consuming from the list")
#         consumed_item = self.x[0]
#         print("Consumed item: ", consumed_item)
#         self.x.remove(consumed_item)
#         self.act = 1

class SAC_Profile():
    policy = "MlpPolicy"
    buffer_size = 1000000
    learning_starts = 100
    learning_rate = 3e-4
    batch_size = 256
    tau = 0.005
    gamma = 0.99
    train_freq = 1
    gradient_steps = 1
    action_noise = None
    ent_coef = "auto"
    target_update_interval = 1
    target_entropy = "auto"
    use_sde = False
    sde_sample_freq = -1
    use_sde_at_warmup = False
    policy_kwargs = None

class BaselineFoundation():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.handler = [None for _  in range(self.n_thread)]
        assert n_agent == 1
        self.num_timesteps = 0
        self.action_noise = [None for _  in range(self.n_thread)]
        self.learning_starts_step = 300
        self.sac = SAC(policy=SAC_Profile.policy)
        self.replay_buffer = ReplayBuffer(trainer_hook=self.sac.train_on_samples)

    def interact_with_env(self, State_Recall):
        last_obs = team_intel['Latest-Obs']
        info = team_intel['Latest-Team-Info']
        done = team_intel['Env-Suffered-Reset']
        step_cnt = team_intel['Current-Obs-Step']
        self.num_timesteps += 1

        # process a env suffering reset:
        for env_index, env_done in enumerate(done):
            if env_done and action_noise[env_index] is not None: action_noise[env_index].reset()

        # action 用于env.step，buffer_action用于buffer
        action, buffer_action = self.sac._sample_action(SAC_Profile.learning_starts, action_noise)

        # traj
        traj_fragment = {'buffer_action':buffer_action, 'last_obs':last_obs, 'can_train':(self.num_timesteps > self.learning_starts_step)}

        # leave a hook to grab the reward signal just a moment later
        State_Recall['_hook_'] = self.replay_buffer.feed_traj(traj_fragment, require_hook=True)  

        action_list = action
        return action_list, State_Recall





class SAC():
    def __init__(self, policy):
        self.policy = policy
        self.target_entropy = SAC_Profile.target_entropy
        self.log_ent_coef = None  # type: Optional[torch.Tensor]
        # Entropy coefficient / Entropy temperature Inverse of the reward scale
        self.ent_coef = SAC_Profile.ent_coef
        self.target_update_interval = SAC_Profile.target_update_interval
        self.ent_coef_optimizer = None
        self._setup_model()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        
    def _setup_model(self) -> None:
        # super(SAC, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically see Automating Entropy Adjustment for Maximum Entropy RL section
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float this will throw an error if a malformed string (different from 'auto') passed
            self.ent_coef_tensor = torch.tensor(float(self.ent_coef)).to(self.device)


    def train_on_samples(self, read_sample_func, gradient_steps: int) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)
        for gradient_step in range(gradient_steps):
            # ！！ 采样
            # Sample replay buffer
            replay_data = read_sample_func(batch_size)
            self.alg_core(replay_data)

        self._n_updates += gradient_steps


    def alg_core(replay_data):
        # We need to sample because `log_std` may have changed between two gradient steps
        if self.use_sde:
            self.actor.reset_noise()

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = torch.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            ent_coef_losses.append(ent_coef_loss.item())
        else:
            ent_coef = self.ent_coef_tensor

        ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

        with torch.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            # Compute the next Q values: min over all critics targets
            next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
        critic_losses.append(critic_loss.item())

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = torch.mean(log_prob - qf1_pi)
        # Mean over all critic networks
        q_values_pi = torch.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
        min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        actor_losses.append(actor_loss.item())

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        if gradient_step % self.target_update_interval == 0:
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)






class ReplayBufferSamples(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray





class ReplayBuffer():

    def __init__(self, trainer_hook, buffer_size):
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float64)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float64)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.long)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False
        self.trainer_hook = trainer_hook

    def feed_traj(self, traj_frag, require_hook=False):
        assert self._traj_lock_buf is None
        # an unlock hook must be exected before new trajectory feed in
        if require_hook: 
            # the traj_frag is not intact, lock up traj_frag, wait for more
            self._traj_lock_buf = traj_frag
            return self._unlock_hook
        else:
            assert 'done' in traj_frag
            self.__batch_update(traj_frag=traj_frag)

    def _unlock_hook(self, traj_frag):
        # now HMP callback with reward and done, as well as new obs
        assert self._traj_lock_buf is not None
        traj_frag.update(self._traj_lock_buf)
        assert 'done' in traj_frag
        self._traj_lock_buf = None
        self._add(
            obs=traj_frag['last_obs'],
            next_obs=traj_frag['new_obs'],
            action=traj_frag['buffer_action'],
            reward=traj_frag['reward'],
            done=traj_frag['reward'],
        )
        if traj_frag['can_train']:
            read_sample_func = lambda: self._sample(self.batch_size)
            self.trainer_hook(read_sample_func)
        return


    def _add(self, obs, next_obs, action, reward, done) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0    # 环形缓冲区？



    def _sample(self, batch_size) -> ReplayBufferSamples:
        if self.full:
            batch_inds = np.random.randint(0, self.buffer_size*self.n_envs, size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.pos*self.n_envs,         size=batch_size)

        data = (self.observations.reshape(-1, observations.shape[-1]          )[batch_inds, :],
                self.actions.reshape(-1, actions.shape[-1]                    )[batch_inds, :],
                self.next_observations.reshape(-1, next_observations.shape[-1])[batch_inds, :],
                self.dones.reshape(-1, dones.shape[-1]                        )[batch_inds],
                self.rewards.reshape(-1, rewards.shape[-1]                    )[batch_inds])

        return ReplayBufferSamples(data)
        