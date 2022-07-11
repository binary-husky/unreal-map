from torch.distributions.categorical import Categorical
import torch
from .foundation import AlgorithmConfig
from UTIL.tensor_ops import repeat_at, _2tensor
from torch.distributions import kl_divergence
EPS = 1e-9
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



def random_process_with_clamp3(probs, hit):
    yita = AlgorithmConfig.yita
    min_prob = AlgorithmConfig.yita_min_prob

    with torch.no_grad():
        max_place = probs.argmax(-1, keepdims=True)
        mask_max = torch.zeros_like(probs).scatter_(dim=-1, index=max_place, value=1).bool()
        pmax = probs[mask_max].reshape(max_place.shape) #probs[max_place].clone()

        # act max
        assert max_place.shape[-1] == 1
        act_max = max_place.squeeze(-1)

        # act samp
        yita_arr = torch.ones_like(pmax)*yita
        # p_hat = pmax + (pmax-1) / (1/yita_arr_clip-1) + 1e-10
        p_hat = (pmax-yita_arr)/((1-yita_arr)+EPS)
        p_hat = p_hat.clamp(min=min_prob)
        k = (1-p_hat)/((1-pmax)+EPS)
        probs *= k
        probs[mask_max] = p_hat.reshape(-1)
        dist = Categorical(probs=probs)
        act_samp = dist.sample()
        assert act_samp.shape[-1] != 1

        hit_e = repeat_at(_2tensor(hit), -1, act_max.shape[-1])

        return torch.where(hit_e, act_max, act_samp)


class CCategorical():
    def __init__(self):
        pass

    def sample(self, dist):
        # p_hit = AlgorithmConfig.yita
        # print('AlgorithmConfig.yita', AlgorithmConfig.yita)
        probs = dist.probs.clone()
        # assert 1/probs.shape[-1] > yita, ('yita is too big, please set it less than', 1/probs.shape[-1])
        # hit = True if torch.rand(()) < p_hit else False
        return random_process_with_clamp3(probs, self.fix_max)
        # return random_process_allow_big_yita(probs, hit)

    def register_fixmax(self, fix_max):
        self.fix_max = fix_max




    def sample_small_yita(self, dist):
        p_hit = AlgorithmConfig.yita
        probs = dist.probs.clone()
        assert 1/probs.shape[-1] > yita, ('yita is too big, please set it less than', 1/probs.shape[-1])
        hit = True if torch.rand(()) < p_hit else False
        return random_process(probs, hit)

    def feed_logits(self, logits):
        try:
            return Categorical(logits=logits)
        except:
            print('error')

    def get_inter_agent_kl_divergence(self, act_dist, logits_agent_cluster):
        # probs = act_dist.probs
        n_agent = logits_agent_cluster.shape[-2]
        # probs_rep = repeat_at(tensor=probs, insert_dim=-2, n_times=n_agent)
        # # probs_rep.S # (?, n_agent, n_agent, n_action)
        # probs_rep_transpose = probs_rep.swapaxes(-2,-3)
        # mat = (probs_rep*probs_rep.log()-probs_rep*probs_rep_transpose.log()).sum(-1)
        
        logits_agent_cluster_expanded = repeat_at(tensor=logits_agent_cluster, insert_dim=-2, n_times=n_agent)
        logits_agent_cluster_expanded_transpose = logits_agent_cluster_expanded.swapaxes(-2,-3)
        cat1 = Categorical(logits=logits_agent_cluster_expanded)
        cat2 = Categorical(logits=logits_agent_cluster_expanded_transpose)
        divergence = kl_divergence(cat1, cat2)
        # assert not torch.isnan(mat).any()
        assert not torch.isnan(divergence).any()

        # return mat
        return divergence