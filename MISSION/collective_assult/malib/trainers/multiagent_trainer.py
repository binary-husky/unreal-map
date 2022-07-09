"""
The trainer for multi-agent training.
"""
import pickle, time, os
from malib.trainers.utils import *


class MATrainer:
    """This class implements a multi-agent trainer.
    """
    def __init__(
            self, env, agents, sampler,
            batch_size=128,
            steps=10000,
            max_path_length=25,
            exploration_steps=100,
            save_after = 25*100,       ## save after every save_after training steps
            training_interval=1,
            extra_experiences=['target_actions'],
            save_path=None,
            is_rommeo=False,
            cont_ckpt=0
    ):
        self.env = env
        self.agents = agents
        self.sampler = sampler
        self.batch_size = batch_size
        self.steps = steps
        self.max_path_length = max_path_length
        self.exploration_steps = exploration_steps
        self.save_after = save_after    ## save after every save_after training steps
        self.training_interval = training_interval
        self.extra_experiences = extra_experiences
        self.losses = []
        self.save_path = save_path
        self.rommeo = is_rommeo
        self.cont_ckpt = cont_ckpt           # ckpt from which to continue training


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

    def run(self):
        for step in range(self.cont_ckpt*self.max_path_length,self.cont_ckpt*self.max_path_length+self.steps):       ## remember that 25 is valid only for
            # print(step)
            if step < self.exploration_steps:                  ## Don't worry about this step for continue training, number of steps will automatically be greater than this
                self.sampler.sample(explore=True)
                continue

            # print('Flag1')
            self.sampler.sample()   # one sample samples one time step
            
            # print('Flag2')
            if step % self.training_interval == 0:
                # print('Flag3')
                batches = self.sample_batches()
                for extra_experience in self.extra_experiences:
                    if extra_experience == 'annealing':
                        batches = add_annealing(batches, step - self.exploration_steps, annealing_scale=1.)
                        # print('annealing', batches[0]['annealing'])
                    elif extra_experience == 'target_actions':
                        batches = add_target_actions(batches, self.agents, self.batch_size)
                    elif extra_experience == 'recent_experiences':
                        batches = add_recent_batches(batches, self.agents, self.batch_size)
                agents_losses = []


                for i, (agent, batch) in enumerate(zip(self.agents, batches)):
                    agent_losses = agent.train(batch)
                    agents_losses.append(agent_losses)
                self.losses.append(agents_losses)
            if step%self.save_after == self.save_after-1:
                 ## test code for saving the model
                self.save(int(step/self.max_path_length))
             

    def save(self, checkpoint):
        print('Saving the agents')
        if self.save_path is None:
            self.save_path = '/tmp/agents.pickle'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        path = self.save_path+'/agents_ckpt_'+str(checkpoint)+'.pickle'
        with open(path, 'wb') as f:
            pickle.dump(self.agents, f, pickle.HIGHEST_PROTOCOL)

    def restore(self):
        path = self.save_path+'/agents_ckpt_'+str(self.cont_ckpt-1)+'.pickle'
        print(path)
        if not os.path.exists(path):
            print('Trained agents don\'t exist for this Config')
            return
        with open(path, 'rb') as f:
            print('Restoring trained agents')
            self.agents = pickle.load(f)
        self.sampler.agents = self.agents
        self.sampler._n_episodes = self.cont_ckpt

    def resume(self):
        #first restore the trained agents
        self.restore()
        self.run()
  
    def log_diagnostics(self):
        pass
