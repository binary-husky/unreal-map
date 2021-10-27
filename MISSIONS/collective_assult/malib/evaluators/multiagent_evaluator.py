"""
The trainer for multi-agent training.
"""
import pickle, time, os
from malib.trainers.utils import *

class MAEvaluator:
    """This class implements a multi-agent trainer.
    """
    def __init__(
            self, env, agents, sampler, ckpt,
            batch_size=128,
            steps=10000,
            extra_experiences=['target_actions'],
            save_path=None,
            is_rommeo=False,
            max_path_length = 25
    ):
        self.env = env
        self.agents = agents
        self.sampler = sampler
        self.batch_size = batch_size
        self.steps = steps
        self.max_path_length = max_path_length
        self.extra_experiences = extra_experiences
        self.losses = []
        self.rommeo = is_rommeo
        self.ckpt = ckpt
        self.save_path = save_path
        self.restore(ckpt)


    def setup(self, env, agents, sampler):
        self.env = env
        self.agents = agents
        self.sampler = sampler

    def sample_batches(self):
        assert len(self.agents) > 1
        batches = []
        indices = self.agents[0].replay_buffer.random_indices(self.batch_size)
        for agent in self.agents:
            batch = agent.replay_buffer.batch_by_indices(indices)
            batches.append(batch)
        return batches

    def do_communication(self):
        pass

    def individual_forward(self):
        pass

    def centralized_forward(self):
        pass

    def apply_gradient(self):
        pass

    def demo(self):         # run single episode of game with rendering     
        # for i in range(10):
        #     current_observation_n = self.env.reset()
        #     for j in range(45):
        #         action_n = []
        #         for agent, current_observation in zip(self.agents, current_observation_n):
        #                 action = agent.act(current_observation.astype(np.float32))
        #                 action_n.append(np.array(action))
        #         next_observation_n, reward_n, done_n, info = self.env.step(action_n)

        #         self.env.render(mode="rgb_array")[0]
        #         time.sleep(0.05)
        #         current_observation_n = next_observation_n
        for i in range(100):
            self.sampler.sample()
            
    def save(self, checkpoint):
        print('Saving the agents')
        if self.save_path is None:
            self.save_path = '/tmp/agents.pickle'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        path = self.save_path+'/agents_ckpt_'+str(checkpoint)+'.pickle'
        with open(path, 'wb') as f:
            pickle.dump(self.agents, f, pickle.HIGHEST_PROTOCOL)

    def restore(self, ckpt):
        path = self.save_path+'/agents_ckpt_'+str(self.ckpt)+'.pickle'
        print(path)
        if not os.path.exists(path):
            print('Trained agents don\'t exist for this Config')
            return
        with open(path, 'rb') as f:
            print('Restoring trained agents')
            self.agents = pickle.load(f)
        self.sampler.agents = self.agents

    def resume(self, ckpt):
        self.restore(ckpt)


    def log_diagnostics(self):
        pass
