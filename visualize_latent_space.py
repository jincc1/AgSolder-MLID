import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from models.wae import WAE
from utils.dataset import FeatureDataset
from utils.utils import get_latents
from utils.plotting import export_to_csv
from config import get_config

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
    dataset = FeatureDataset(raw_x, raw_y)

    # Load model
    model_dir = os.path.join(root, '{}/{}_{}.pth'.format(params['model_name'], params['model_name'], params['num_epochs']))
    model = WAE(raw_x.shape[1]).to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    # --- Latent Space Visualization ---
    latents = get_latents(model, dataset, device)

    # Export subsets of latent space to CSV
    high_ag = ((raw_x[:, 0] > 0.3)
    high_ag_latent = latents[high_ag]
    high_ag_color = raw_y[:][high_ag]
    export_to_csv(high_ag_latent, high_ag_color, 'high_ag_data.csv')
    
    high_cu = (raw_x[:, 1] > 0.3)
    high_cu_latent = latents[high_cu]
    high_cu_color = raw_y[:][high_cu]
    export_to_csv(high_cu_latent, high_cu_color, 'high_cu_data.csv')
    
    high_in = (raw_x[:, 2] > 0.3)
    high_in_latent = latents[high_in]
    high_in_color = raw_y[:][high_in]
    export_to_csv(high_in_latent, high_in_color, 'high_in_data.csv')
    
    
    # Plot settings for latent space visualization
    fig, axs = plt.subplots(figsize=(4, 4), dpi=500)
    axs.set_yticks(np.arange(-6, 8, step=2))
    axs.set_xticks(np.arange(-10, 5, step=2))
    axs.set_yticklabels(np.arange(-6, 8, step=2), fontsize=7)
    axs.set_xticklabels(np.arange(-10, 5, step=2), fontsize=7)
    
    for axis in ['top', 'bottom', 'left', 'right']:
        axs.spines[axis].set_linewidth(1.)
    
    axs.tick_params(axis='both', which='major', top=False, labeltop=False, direction='out', width=1., length=4)
    axs.tick_params(axis='both', which='major', right=False, labelright=False, direction='out', width=1., length=4)
    
    # Scatter plots of different alloy groups
    scatter1 = axs.scatter(high_ag_latent[:, 0], high_ag_latent[:, 1], c='blue', alpha=.55, s=8, linewidths=0, label='Alloys  Ag')
    scatter2 = axs.scatter(high_cu_latent[:, 0], high_cu_latent[:, 1], c='yellow', alpha=.65, s=14, linewidths=0, label='Alloys  Cu')
    scatter3 = axs.scatter(high_in_latent[:, 0], high_in_latent[:, 1], c='red', alpha=.65, s=14, linewidths=0, marker='s', label='Alloys w/o In')

    handles, labels = axs.get_legend_handles_labels()
    handles = handles[::1]
    labels = labels[::1]

    legend_properties = {'size': 7.5}
    axs.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.015, 1.017), handletextpad=-0.3, frameon=False, prop=legend_properties)

    fig.savefig(os.path.join(root, 'latent_space_visualization.png'), bbox_inches='tight', pad_inches=0.01)