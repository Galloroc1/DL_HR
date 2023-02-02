import torch
from data import get_torch_dataloader
from model import MLPDiffusion, DiffusionModel
from params import Diff_params
from tqdm import tqdm
from function import p_sample_loop
from utils import show_img

train_params = {
    "num_steps": 100,
    "batch_size": 128,
    "epoch": 4000,
    'lr': 0.001
}

# params like alpha,beta..
diffusion_params = Diff_params(train_params['num_steps']).get_params()
dataloader = get_torch_dataloader(batch_size=train_params['batch_size'])
model = MLPDiffusion(train_params['num_steps']).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
loss_func = DiffusionModel()

for t in tqdm(range(train_params['epoch'])):
    for idx, batch_x in enumerate(dataloader):
        loss = loss_func.forward(model, batch_x.cuda(), diffusion_params)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    if t % 100 == 0:
        print(loss)
        x_seq = p_sample_loop(model, dataloader.dataset.shape,
                              train_params['num_steps'], diffusion_params.betas,
                              diffusion_params.one_minus_alphas_bar_sqrt)
        show_img(x_seq, t)
