import numpy as np
import copy
import math
import random

class ExampleFoundation():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.handler = [None for _  in range(self.n_thread)]

    def interact_with_env(self, team_intel):
        info = team_intel['Latest-Team-Info']
        done = team_intel['Env-Suffered-Reset']
        step_cnt = team_intel['Current-Obs-Step']
        action_list = np.zeros(shape=(self.n_agent, self.n_thread, 1))
        return action_list, team_intel


