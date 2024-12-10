import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import os
import pandas as pd

def weights_init(m):
    """
    Initializes weights in the neural network.
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def set_seed(seed):
    """
    Sets random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_latents(model, dataset, device):
    """
    Extracts latent space representations from the dataset using the model.
    """
    model.to(device).eval()
    latents = []
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        for i, data in enumerate(dataloader):
            x = data[0].to(device)
            recon_x, z = model(x)
            latents.append(z.detach().cpu().numpy())
    return np.concatenate(latents, axis=0)

def imq_kernel(X: torch.Tensor, Y: torch.Tensor, h_dim: int, device):
    """
    Computes the IMQ (Inverse Multi-Quadratic) kernel between two tensors.
    """
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)
    prods_x = torch.mm(X, X.t())
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)
    prods_y = torch.mm(Y, Y.t())
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).to(device)) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats