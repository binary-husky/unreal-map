from torch.distributions.categorical import Categorical
import torch
from .foundation import AlgorithmConfig
# yita = p_hit = 0.14

def random_process(probs, hit):
    yita = AlgorithmConfig.yita
    with torch.no_grad():
        max_place = probs.argmax(-1, keepdims=True)
        mask_max = torch.zeros_like(probs).scatter_(-1, max_place, 1).bool()
        pmax = probs[mask_max]
        if hit:
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

def random_process_allow_big_yita(probs, hit):
    yita = AlgorithmConfig.yita
    with torch.no_grad():
        max_place = probs.argmax(-1, keepdims=True)
        mask_max = torch.zeros_like(probs).scatter_(-1, max_place, 1).bool()
        pmax = probs[mask_max].reshape(max_place.shape) #probs[max_place].clone()
        if hit:
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


class CCategorical():
    def __init__(self):
        pass

    def sample(self, dist):
        p_hit = AlgorithmConfig.yita
        probs = dist.probs.clone()
        # assert 1/probs.shape[-1] > yita, ('yita is too big, please set it less than', 1/probs.shape[-1])
        hit = True if torch.rand(()) < p_hit else False
        return random_process_allow_big_yita(probs, hit)



    def sample_small_yita(self, dist):
        p_hit = AlgorithmConfig.yita
        probs = dist.probs.clone()
        assert 1/probs.shape[-1] > yita, ('yita is too big, please set it less than', 1/probs.shape[-1])
        hit = True if torch.rand(()) < p_hit else False
        return random_process(probs, hit)

    def feed_logits(self, logits):
        return Categorical(logits=logits)
