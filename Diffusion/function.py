import torch


def get_qxt(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
    """可以基于x[0]得到任意时刻t的x[t]"""
    noise = torch.randn_like(x_0).cuda()
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return x_0 * alphas_t + noise * alphas_1_m_t, noise


def p_sample_loop(model,
                  shape,
                  n_steps,
                  betas,
                  one_minus_alphas_bar_sqrt):
    # Step7 逆扩散采样函数（inference）
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""

    cur_x = torch.randn(shape).cuda()
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model,
             x,
             t,
             betas,
             one_minus_alphas_bar_sqrt):
    """从x[T]采样t时刻的重构值"""

    t = torch.tensor([t]).cuda()
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
    z = torch.randn_like(x).cuda()
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample
