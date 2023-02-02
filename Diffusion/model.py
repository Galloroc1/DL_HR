import torch.nn as nn
import torch
from function import get_qxt


class MLPDiffusion(nn.Module):

    def __init__(self, n_steps, num_units=128):
        super().__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2),
            ]
        )

        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x, t):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
        return x


class DiffusionModel:

    def __init__(self):
        pass

    def forward(self, model, x_0, diffusion_params):
        batch_size = x_0.shape[0]
        # get t and step - t
        t = torch.randint(0, diffusion_params.num_steps, size=(batch_size // 2,))
        t = torch.cat([t, diffusion_params.num_steps - 1 - t], dim=0)
        t = t.unsqueeze(-1).cuda()

        x, noise = get_qxt(x_0, t, diffusion_params.alphas_bar_sqrt, diffusion_params.one_minus_alphas_bar_sqrt)
        output = model(x, t.squeeze(-1))
        return torch.nn.MSELoss()(noise, output)

        # return (noise - output).square().mean()
