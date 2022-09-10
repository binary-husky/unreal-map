from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from UTIL.tensor_ops import my_view

class HyperNet(nn.Module):
    def __init__(self, **kwargs):
        super(HyperNet, self).__init__()

        self.x_input_dim = kwargs['x_input_dim']
        self.embed_dim = kwargs['embed_dim']
        self.hyper_input_dim = kwargs['hyper_input_dim']
        
        # hyper w1 b1
        self.hyper_w1 = nn.Sequential( nn.Linear(self.hyper_input_dim, self.embed_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.embed_dim, self.x_input_dim * self.embed_dim))
        
        self.hyper_b1 = nn.Sequential(nn.Linear(self.hyper_input_dim, self.embed_dim))
        
        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(
                            nn.Linear(self.hyper_input_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, self.embed_dim * self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.hyper_input_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))

    def forward(self, x, hyper_x):
        # x shape (thread/batch, agent, core)
        # hyper_x shape (thread/batch, core)
        assert hyper_x.dim() == 3
        # reshape w1 into # (..., x_input_dim, embed_dim)
        w1 = my_view(self.hyper_w1(hyper_x), [0, 0, self.x_input_dim, self.embed_dim])
        b1 = self.hyper_b1(hyper_x).unsqueeze(-2) # b1 (thread/batch, core=embed_dim)
        
        # Second layer
        w2 = my_view(self.hyper_w2(hyper_x), [0, 0, self.embed_dim, self.embed_dim])
        b2 = self.hyper_b2(hyper_x).unsqueeze(-2)
        
        ## x shape = (..., x_input_dim)
        ## w1 shape = (..., x_input_dim, embed_dim)
        
        # x reshape = (..., 1, x_input_dim)
        x = x.unsqueeze(-2)
        hidden = F.elu(torch.matmul(x, w1) + b1) # b * t, 1, emb
        # Forward (batch, 1, 32) * w2(batch, 32, 1) => y(batch, 1)
        y = torch.matmul(hidden, w2) + b2 # b * t, 1, 1
        
        return y.squeeze(-2)