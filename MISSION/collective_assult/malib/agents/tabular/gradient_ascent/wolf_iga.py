# Created by yingwen at 2019-03-10

from malib.agents.tabular.utils import V

def WoLF_IGA(pi_alpha,
             pi_beta,
             payoff_0,
             payoff_1,
             u_alpha,
             u_beta,
             iteration=500,
             pi_alpha_nash=0.5,
             pi_beta_nash=0.5,
             lr_min=0.01,
             lr_max=0.04):
    pi_alpha_history = [pi_alpha]
    pi_beta_history = [pi_beta]
    pi_alpha_gradient_history = [0.]
    pi_beta_gradient_history = [0.]
    for i in range(iteration):
        lr_alpha = lr_max
        lr_beta = lr_max
        if V(pi_alpha, pi_beta, payoff_0) > V(pi_alpha_nash, pi_beta, payoff_0):
            lr_alpha = lr_min
        if V(pi_alpha, pi_beta, payoff_1) > V(pi_alpha, pi_beta_nash, payoff_0):
            lr_beta = lr_min

        pi_alpha_gradient = (pi_beta * u_alpha + payoff_0[(0, 1)] - payoff_0[(1, 1)])
        pi_beta_gradient = (pi_alpha * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)])
        pi_alpha_gradient_history.append(pi_alpha_gradient)
        pi_beta_gradient_history.append(pi_beta_gradient)
        pi_alpha_next = pi_alpha + lr_alpha * pi_alpha_gradient
        pi_beta_next = pi_beta + lr_beta * pi_beta_gradient
        pi_alpha = max(0., min(1., pi_alpha_next))
        pi_beta = max(0., min(1., pi_beta_next))
        pi_alpha_history.append(pi_alpha)
        pi_beta_history.append(pi_beta)
    return pi_alpha_history, \
           pi_beta_history, \
           pi_alpha_gradient_history, \
           pi_beta_gradient_history