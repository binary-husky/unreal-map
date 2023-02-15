import os, time, torch, traceback
import numpy as np
from config import GlobalConfig
from UTIL.colorful import *

class AlgorithmBase():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.team = team
        self.act_space = space['act_space']
        self.obs_space = space['obs_space']
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        self.mcv = mcv
        self.device = GlobalConfig.device

    def interact_with_env(self, team_intel):
        raise NotImplementedError
