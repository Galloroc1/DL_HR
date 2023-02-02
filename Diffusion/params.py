import torch


class Diff_params:
    def __init__(self, num_steps=100):
        self.betas = None
        self.one_minus_alphas_bar_sqrt = None
        self.one_minus_alphas_bar_log = None
        self.alphas_bar_sqrt = None
        self.alphas_prod_p = None
        self.alphas_prod = None
        self.alphas = None
        self.num_steps = num_steps

    def get_params(self):
        # 计算alpha, alpha_prod, alpha_prod_previous, alpha_bar_sqrt等变量值
        betas = torch.linspace(-6, 6, self.num_steps).cuda()
        self.betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        self.alphas = 1 - self.betas

        self.alphas_prod = torch.cumprod(self.alphas, 0)

        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
        return self
