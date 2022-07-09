# Created by yingwen at 2019-03-11

def IGA_TR(pi_alpha,
          pi_beta,
          payoff_0,
          payoff_1,
          u_alpha,
          u_beta,
          iteration=200,
          lr=.1):

    pi_alpha_history = [pi_alpha]
    pi_beta_history = [pi_beta]
    pi_alpha_gradient_history = [0.]
    pi_beta_gradient_history = [0.]
    for i in range(iteration):
        pi_alpha_gradient = (pi_beta * u_alpha + payoff_0[(0, 1)] - payoff_0[(1, 1)])
        pi_beta_gradient = (pi_alpha * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)])
        pi_alpha_gradient_history.append(pi_alpha_gradient)
        pi_beta_gradient_history.append(pi_beta_gradient)
        pi_alpha_new = pi_alpha + lr * pi_alpha_gradient
        pi_beta_new = pi_beta + lr * pi_beta_gradient
        pi_alpha_adv_gradient = (pi_beta_new - pi_beta) * u_alpha * (1. + lr)
        pi_beta_adv_gradient = (pi_alpha_new - pi_alpha) * u_beta * (1. + lr)
        print((pi_alpha_gradient + pi_alpha_adv_gradient))
        pi_alpha_next = pi_alpha + lr * (pi_alpha_gradient + pi_alpha_adv_gradient)
        pi_beta_next = pi_beta + lr * (pi_beta_gradient + pi_beta_adv_gradient)
        pi_alpha = max(0., min(1., pi_alpha_next))
        pi_beta = max(0., min(1., pi_beta_next))
        pi_alpha_history.append(pi_alpha)
        pi_beta_history.append(pi_beta)
    return pi_alpha_history, \
           pi_beta_history, \
           pi_alpha_gradient_history, \
           pi_beta_gradient_history
