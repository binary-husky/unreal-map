import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import DynamicNorm

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, use_normalization=False):
        super().__init__()
        activation_func = nn.ReLU
        h_dim = hidden_dim
        if use_normalization:
            print('test DynamicNorm')
            self.mlp = nn.Sequential(
                DynamicNorm(in_dim, only_for_last_dim=True, exclude_one_hot=True),
                nn.Linear(in_dim, h_dim),
                activation_func(inplace=True),
                nn.Linear(h_dim, out_dim)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, h_dim),
                activation_func(inplace=True),
                nn.Linear(h_dim, out_dim)
            )
    def forward(self,x):
        return self.mlp(x)

class ResLinear(nn.Module):
    def __init__(self, io_dim, h_dim, need_input_tf=False, input_tf_dim=None, inplace_relu=True) -> None:
        super(ResLinear, self).__init__()
        self.need_input_tf = need_input_tf
        if need_input_tf:
            self.f0 = nn.Linear(input_tf_dim, io_dim)
        self.f1 = nn.Linear(io_dim, h_dim)
        self.lkrelu = nn.ReLU(inplace=True) if inplace_relu else nn.ReLU(inplace=False)
        self.f2 = nn.Linear(h_dim, io_dim)

    def forward(self, xo):
        if self.need_input_tf:
            xo = self.f0(xo)
        x = self.lkrelu(self.f1(xo))
        x = self.f2(x) + xo
        x = self.lkrelu(x)
        return x
        
class LinearFinal(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(LinearFinal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )