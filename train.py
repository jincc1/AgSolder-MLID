import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.wae import WAE
from utils.dataset import FeatureDataset
from utils.utils import set_seed, imq_kernel
from config import get_config

def train_WAE(model, optimizer, dataloader, params, device, root):
    """
    Trains the WAE model.
    """
    model_name = params['model_name']
    num_epochs = params['num_epochs']
    sigma = params['sigma']
    MMD_lambda = params['mmd_lambda']

    folder_dir = os.path.join(root, model_name)
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)
    loss_ = []
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = []  # save for plot, recon loss + MMD
        total_recon = []  # binary cross entropy
        total_MMD = []  # maximum mean discrepancy

        for i, data in enumerate(dataloader):
            x = data[0].to(device)
            y = data[1].to(device)
            model.train()  # Set model to training mode
            recon_x, z_tilde = model(x)
            z = sigma * torch.randn(z_tilde.size()).to(device)

            # Reconstruction loss
            recon_loss = F.l1_loss(recon_x, x, reduction='mean')

            # MMD loss
            MMD_loss = imq_kernel(z_tilde, z, h_dim=2, device=device)
            MMD_loss = MMD_loss / x.size(0)  # Averaging, because recon loss is mean.
            loss = recon_loss + MMD_loss * MMD_lambda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            total_recon.append(recon_loss.item())
            total_MMD.append(MMD_loss.item())

        avg_loss = sum(total_loss) / len(total_loss)
        avg_recon = sum(total_recon) / len(total_recon)
        avg_MMD = sum(total_MMD) / len(total_MMD)
        loss_.append(avg_loss)

        output_message = '[{:03}/{:03}] loss: {:.6f} Recon_loss: {:.6f}, MMD_loss:{:.6f}, time: {:.3f} sec\n'.format(
            epoch + 1, num_epochs, avg_loss, avg_recon, avg_MMD, time.time() - start_time)

        # Print the message
        print(output_message)

        # Save the print output to the file
        with open('print_output.txt', 'a') as file:
            file.write(output_message)
        # save the model every 5 epoches
        if (epoch + 1) % 5 == 0:
            save_model_dir = str(model_name + "_{}.pth".format(epoch + 1))
            torch.save(model.state_dict(), os.path.join(folder_dir, save_model_dir))
    return loss_


if __name__ == '__main__':
    # Get configuration
    config = get_config()
    params = config.params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = config.root

    # Load data
    all_data = pd.read_csv(config.data_path, header=0).iloc[:, 0:18].to_numpy()
    raw_x = all_data[:128, [0, 1, 2, 5]]
    raw_y = all_data[:128, -1].reshape(-1, 1)

    # Create dataset and dataloader
    dataset = FeatureDataset(raw_x, raw_y)
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    # Set seed
    set_seed(config.seed)

    # Initialize model and optimizer
    model = WAE(raw_x.shape[1]).to(device)
    optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # Train the model
    loss_ = train_WAE(model, optimizer, dataloader, params, device, root)

    # Plot the training loss
    plt.figure()
    sns.set_style('ticks')
    plt.plot(range(len(loss_)), loss_)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig(os.path.join(root, params['model_name'], 'loss_curve.png'))
    plt.show()