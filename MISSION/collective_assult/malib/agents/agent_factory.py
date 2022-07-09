# Created by yingwen at 2019-06-30

from malib.agents.ddpg.ddpg import DDPGAgent
from malib.agents.ddpg.maddpg import MADDPGAgent
from malib.agents.ddpg.ddpg_om import DDPGOMAgent
from malib.agents.ddpg.ddpg_tom import DDPGToMAgent
from malib.agents.rommeo.rommeo import ROMMEOAgent
# from malib.agents.rommeo.rommeo_cheat import CheatROMMEOAgent
# from malib.agents.rommeo.rommeo_cheat_stable import CheatStableROMMEOAgent
# from malib.agents.rommeo.rommeo_fake import FakeROMMEOAgent
# from malib.agents.rommeo.adhoc_rommeo import AdhocROMMEOAgent
# from malib.agents.rommeo.rommeo_mse import ROMMEOMSEAgent
from malib.agents.sac.sac import SACAgent
from malib.agents.gr2.pr2 import PR2Agent
from malib.agents.gr2.pr2_soft import PR2SoftAgent
from malib.agents.gr2.pr2k import PR2KSoftAgent
from malib.policies import DeterministicMLPPolicy, GaussianMLPPolicy, RelaxedSoftmaxMLPPolicy
from malib.value_functions import MLPValueFunction
from malib.replay_buffers import IndexedReplayBuffer
from malib.replay_buffers import IndexedRolloutReplayBuffer
from malib.policies.explorations.ou_exploration import OUExploration
# from examples.third_party.nn import GaussianPolicy


def get_ddpg_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size, policy_type='dete'):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    if policy_type == 'dete':
        policy_fn = DeterministicMLPPolicy
        exploration_strategy = OUExploration(action_space)
    elif policy_type == 'gumble':
        policy_fn = RelaxedSoftmaxMLPPolicy
        exploration_strategy = None
    return DDPGAgent(
        env_specs=env.env_specs,
        policy=policy_fn(
            input_shapes=(observation_space.shape,),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, action_space.shape),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=exploration_strategy,
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_sac_agent(env, hidden_layer_sizes,
                  max_replay_buffer_size, policy_type='gaussian'):
    """
    SAC agent for single player learning.
    """
    observation_space = env.env_specs.observation_space[0]
    action_space = env.env_specs.action_space[0]
    env_specs = env.env_specs
    if policy_type == 'gaussian':
        policy_fn = GaussianMLPPolicy
    elif policy_type == 'gumble':
        policy_fn = RelaxedSoftmaxMLPPolicy

    return SACAgent(
        env_specs=env_specs,
        policy=policy_fn(
            input_shapes=observation_space.shape,
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='{}_policy'.format(policy_type),
        ),
        qfs=[MLPValueFunction(
            input_shapes=(observation_space.shape, action_space.shape),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_{}'.format(qf_id)
        )
            for qf_id in range(2)],
        vf=MLPValueFunction(
            input_shapes=(observation_space.shape,),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='vf'
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          max_replay_buffer_size=max_replay_buffer_size,
                                          )
    )




def get_rommeo_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size, policy_type = 'gaussian', uniform=False, custom_b=False, bi=1., bj=1.):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    opponent_action_shape = (env.env_specs.action_space.opponent_flat_dim(agent_id),)
    if policy_type == 'gaussian':
        policy_fn = GaussianMLPPolicy
    elif policy_type == 'gumble':
        policy_fn = RelaxedSoftmaxMLPPolicy
    return ROMMEOAgent(
        env_specs=env.env_specs,
        policy=policy_fn(
            input_shapes=(observation_space.shape, opponent_action_shape),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id),
            repara=True,
            # smoothing_coefficient=0.5
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, action_space.shape, opponent_action_shape),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          max_replay_buffer_size=max_replay_buffer_size,
                                          opponent_action_dim=opponent_action_shape[0],
                                          ),
        opponent_policy=policy_fn(
            input_shapes=(observation_space.shape,),
            output_shape=opponent_action_shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='opponent_policy_agent_{}'.format(agent_id),
            repara=True,
        ),
        gradient_clipping=10,
        agent_id=agent_id,
        name='ROMMEO_{}'.format(agent_id),
        uniform=uniform, custom_b=custom_b, bi=bi, bj=bj
    )


def get_pr2_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size, policy_type ='deter'):
    print(agent_id)
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    opponent_action_shape = (env.env_specs.action_space.opponent_flat_dim(agent_id),)
    print(opponent_action_shape, 'opponent_action_shape')
    if policy_type == 'dete':
        policy_fn = DeterministicMLPPolicy
        exploration_strategy = OUExploration(action_space)
    elif policy_type == 'gumble':
        policy_fn = RelaxedSoftmaxMLPPolicy
        exploration_strategy = None
    return PR2Agent(
        env_specs=env.env_specs,
        policy=policy_fn(
            input_shapes=(observation_space.shape,),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, action_space.shape, opponent_action_shape),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        ind_qf=MLPValueFunction(
            input_shapes=(observation_space.shape, action_space.shape),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='ind_qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          max_replay_buffer_size=max_replay_buffer_size,
                                          opponent_action_dim=opponent_action_shape[0],
                                          ),
        opponent_policy=policy_fn(
            input_shapes=(observation_space.shape, action_space.shape),
            output_shape=opponent_action_shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='opponent_policy_agent_{}'.format(agent_id)
        ),
        exploration_strategy=exploration_strategy,
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_pr2_soft_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size, policy_type = 'gaussian'):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    opponent_action_shape = (env.env_specs.action_space.opponent_flat_dim(agent_id),)
    if policy_type == 'gaussian':
        policy_fn = GaussianMLPPolicy
    elif policy_type == 'gumble':
        policy_fn = RelaxedSoftmaxMLPPolicy
    return PR2SoftAgent(
        env_specs=env.env_specs,
        policy=policy_fn(
            input_shapes=(observation_space.shape,),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, action_space.shape, opponent_action_shape),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          max_replay_buffer_size=max_replay_buffer_size,
                                          opponent_action_dim=opponent_action_shape[0],
                                          ),
        opponent_policy=policy_fn(
            input_shapes=(observation_space.shape, action_space.shape),
            output_shape=opponent_action_shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='opponent_policy_agent_{}'.format(agent_id)
        ),
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_pr2k_soft_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size, k=2, mu=0):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    opponent_action_shape = (env.env_specs.action_space.opponent_flat_dim(agent_id),)
    print(opponent_action_shape, 'opponent_action_shape')
    return PR2KSoftAgent(
        env_specs=env.env_specs,
        main_policy=GaussianMLPPolicy(
            input_shapes=(observation_space.shape, opponent_action_shape),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        opponent_policy=GaussianMLPPolicy(
            input_shapes=(observation_space.shape, action_space.shape),
            output_shape=opponent_action_shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='opponent_policy_agent_{}'.format(agent_id)
        ),
        prior_policy=GaussianMLPPolicy(
            input_shapes=(observation_space.shape,),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='prior_policy_agent_{}'.format(agent_id)
        ),
        opponent_prior_policy=GaussianMLPPolicy(
            input_shapes=(observation_space.shape,),
            output_shape=opponent_action_shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='opponent_prior_policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, action_space.shape, opponent_action_shape),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          max_replay_buffer_size=max_replay_buffer_size,
                                          opponent_action_dim=opponent_action_shape[0],
                                          ),
        k=k,
        mu=mu,
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_maddpg_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    return MADDPGAgent(
        env_specs=env.env_specs,
        policy=DeterministicMLPPolicy(
            input_shapes=(observation_space.shape,),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, (env.env_specs.action_space.flat_dim,)),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          opponent_action_dim=env.env_specs.action_space.opponent_flat_dim(agent_id),
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_ddpgom_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    return DDPGOMAgent(
        env_specs=env.env_specs,
        policy=DeterministicMLPPolicy(
            input_shapes=(observation_space.shape,),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, (env.env_specs.action_space.flat_dim,)),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        opponent_policy=DeterministicMLPPolicy(
            input_shapes=(observation_space.shape,),
            output_shape=(env.env_specs.action_space.opponent_flat_dim(agent_id),),
            hidden_layer_sizes=hidden_layer_sizes,
            name='opponent_policy_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          opponent_action_dim=env.env_specs.action_space.opponent_flat_dim(agent_id),
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_ddpgtom_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    return DDPGToMAgent(
        env_specs=env.env_specs,
        policy=DeterministicMLPPolicy(
            input_shapes=(observation_space.shape, (env.env_specs.action_space.opponent_flat_dim(agent_id),)),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, (env.env_specs.action_space.flat_dim,)),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        opponent_policy=DeterministicMLPPolicy(
            input_shapes=(observation_space.shape,),
            output_shape=(env.env_specs.action_space.opponent_flat_dim(agent_id),),
            hidden_layer_sizes=hidden_layer_sizes,
            name='opponent_policy_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          opponent_action_dim=env.env_specs.action_space.opponent_flat_dim(agent_id),
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )