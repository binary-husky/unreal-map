from torch.distributions.categorical import Categorical
import torch
from .foundation import AlgorithmConfig
from UTIL.tensor_ops import repeat_at, _2tensor
from torch.distributions import kl_divergence
EPS = 1e-9
# yita = p_hit = 0.14

def random_process(probs, rsn_flag):
    yita = AlgorithmConfig.yita
    with torch.no_grad():
        max_place = probs.argmax(-1, keepdims=True)
        mask_max = torch.zeros_like(probs).scatter_(-1, max_place, 1).bool()
        pmax = probs[mask_max]
        if rsn_flag:
            assert max_place.shape[-1] == 1
            return max_place.squeeze(-1)
        else:
            # forbit max prob being chosen, pmax = probs.max(axis=-1)
            p_hat = pmax + (pmax-1)/(1/yita-1)
            k = 1/(1-yita)
            #!!! write
            probs *= k  
            #!!! write
            probs[mask_max] = p_hat 
            # print(probs)
            dist = Categorical(probs=probs)
            samp = dist.sample()
            assert samp.shape[-1] != 1
            return samp

def random_process_allow_big_yita(probs, rsn_flag):
    yita = AlgorithmConfig.yita
    with torch.no_grad():
        max_place = probs.argmax(-1, keepdims=True)
        mask_max = torch.zeros_like(probs).scatter_(-1, max_place, 1).bool()
        pmax = probs[mask_max].reshape(max_place.shape) #probs[max_place].clone()
        if rsn_flag:
            assert max_place.shape[-1] == 1
            return max_place.squeeze(-1)
        else:
            # forbit max prob being chosen
            # pmax = probs.max(axis=-1) #probs[max_place].clone()
            yita_arr = torch.ones_like(pmax)*yita
            yita_arr_clip = torch.minimum(pmax, yita_arr)
            # p_hat = pmax + (pmax-1) / (1/yita_arr_clip-1) + 1e-10
            p_hat = (pmax-yita_arr_clip)/(1-yita_arr_clip)
            k = 1/(1-yita_arr_clip)
            probs *= k
            probs[mask_max] = p_hat.reshape(-1)

            # print(probs)
            dist = Categorical(probs=probs)
            samp = dist.sample()
            assert samp.shape[-1] != 1
            return samp #.squeeze(-1)



def random_process_with_clamp3(probs, yita, yita_min_prob, rsn_flag):

    with torch.no_grad():
        max_place = probs.argmax(-1, keepdims=True)
        mask_max = torch.zeros_like(probs).scatter_(dim=-1, index=max_place, value=1).bool()
        pmax = probs[mask_max].reshape(max_place.shape)
        # act max
        assert max_place.shape[-1] == 1
        act_max = max_place.squeeze(-1)
        # act samp
        yita_arr = torch.ones_like(pmax)*yita
        # p_hat = pmax + (pmax-1) / (1/yita_arr_clip-1) + 1e-10
        p_hat = (pmax-yita_arr)/((1-yita_arr)+EPS)
        p_hat = p_hat.clamp(min=yita_min_prob)
        k = (1-p_hat)/((1-pmax)+EPS)
        probs *= k
        probs[mask_max] = p_hat.reshape(-1)
        dist = Categorical(probs=probs)
        act_samp = dist.sample()
        # assert act_samp.shape[-1] != 1
        hit_e = _2tensor(rsn_flag)
        return torch.where(hit_e, act_max, act_samp)


class CCategorical():
    def __init__(self, planner):
        self.planner = planner
        
        pass

    def sample(self, dist, eprsn):
        probs = dist.probs.clone()
        return random_process_with_clamp3(probs, self.planner.yita, self.planner.yita_min_prob, eprsn)

    def register_rsn(self, rsn_flag):
        self.rsn_flag = rsn_flag

    def feed_logits(self, logits):
        try:
            return Categorical(logits=logits)
        except:
            print('error')
