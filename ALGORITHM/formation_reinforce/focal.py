from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, # the optional argument weight should be a 1D Tensor assigning weight to each of the classes
            reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'


    # x is expected to contain raw, unnormalized scores for each class
    # y is expected to contain class labels, not onehot! e.g [1,5,3,2]
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            assert False, ('I do not use this')
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = (y != self.ignore_index)   #  class label to ignore. Defaults to -100
        y = y[unignored_mask]   # delete samples with ignore index
        if len(y) == 0: # nothing to process
            return 0.
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl


'''
<test 03>: focal loss

reward_est_flatten = torch.Tensor([[0.1,0.2,0.3],[-0.1,-0.2,-0.3]])
reward_class_flatten = torch.Tensor([0,2])
似然概率p = torch.softmax(reward_est_flatten,-1) =
          类别1   类别2    类别3
tensor([[0.3006, 0.3322, 0.3672],   样本1
        [0.3672, 0.3322, 0.3006]])  样本2


对数似然概率p = torch.log(torch.softmax(reward_est_flatten,-1)) =
          类别1   类别2    类别3
tensor([[-1.2019, -1.1019, -1.0019],      样本1
        [-1.0019, -1.1019, -1.2019]])     样本2


==============================================
计算交叉熵  
F.cross_entropy(input=reward_est_flatten, target=reward_class_flatten.long())

math:   # https://www.cnblogs.com/jclian91/p/9376117.html

(1/样本数N)  * 所有样本加和Sigma_N{   
                            所有类别加和Sigma_C{ 
                                -单热标签y_(样本n, 类别c)*log(似然概率p_(样本n,c)) 
                            }           
                        } = 1.2019 == tensor(1.2019)

==============================================
计算focal交叉熵

from .focal import FocalLoss
focal_loss = FocalLoss(alpha=None, gamma=2)
focal_loss(x=reward_est_flatten, y=reward_class_flatten.long())
math:  (1-0.3006)**2 * 1.2019 == tensor(0.5879)

============================================== weight
focal_loss_weighted = FocalLoss(alpha=torch.Tensor([1,2,3]), gamma=2)
focal_loss_weighted(x=reward_est_flatten, y=reward_class_flatten.long())
math:  ((1-0.3006)**2 * 1.2019*(weight=1) + (1-0.3006)**2 * 1.2019*(weight=3)) / (n_sample=2) == tensor(1.1759)


============================================== ignore
focal_loss_weighted_ignore = FocalLoss(alpha=torch.Tensor([1,2,3]), gamma=2, ignore_index=0)
focal_loss_weighted_ignore(x=reward_est_flatten, y=reward_class_flatten.long())
math:  ((1-0.3006)**2 * 1.2019*(weight=1)*(ignore=0) + (1-0.3006)**2 * 1.2019*(weight=3)) / (n_sample=1) == tensor(1.7638)

math:   (1/样本数N)  * 所有样本加和Sigma_N{   
                            所有类别加和Sigma_C{ 
                                -单热标签y_(样本n, 类别c)*
                                    {
                                        log(似然概率p_(n,c))*(1-似然概率p_(n,c))^γ
                                    }
                            }           
                        } = 1.2019 == tensor(1.2019)

'''
