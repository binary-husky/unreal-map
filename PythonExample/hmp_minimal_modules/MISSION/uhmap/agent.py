import numpy as np
from .actset_lookup import agent_json2local_attrs

class Agent(object):
    def __init__(self, team, team_id, uid) -> None:
        self.team = team
        self.team_id = team_id
        self.uid = uid
        self.attrs = agent_json2local_attrs
        for attr_json, attr_agent in self.attrs: setattr(self, attr_agent, None)
        self.pos3d = np.array([np.nan, np.nan, np.nan])
        self.pos2d = np.array([np.nan, np.nan])
    
    def update_agent_attrs(self, dictionary):
        if (not dictionary['agentAlive']):
            self.alive = False
        else:
            assert dictionary['valid']
            for attr_json, attr_agent in self.attrs: 
                setattr(self, attr_agent, dictionary[attr_json])
            assert self.uid == self.uid_remote
            self.pos3d = np.array(self.location)
            self.pos2d = self.pos3d[:2]
            self.vel3d = np.array(self.velocity)
            self.vel2d = self.pos3d[:2]
            self.scale3d = np.array(self.scale3)
            self.scale = self.scale3[0]
            self.yaw = self.rotation[0]

