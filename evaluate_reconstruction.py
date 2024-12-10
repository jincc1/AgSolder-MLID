import torch
import pandas as pd
import os
from models.wae import WAE
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

    # Load model
    model_dir = os.path.join(root, '{}/{}_{}.pth'.format(params['model_name'], params['model_name'], params['num_epochs']))
    model = WAE(raw_x.shape[1]).to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    # --- Reconstruction Check ---
    with torch.no_grad():
        test = torch.FloatTensor(raw_x).to(device)
        recon_x, z = model(test)
        recon_x = model.decoder(z)
        recon_x = recon_x.cpu().detach().numpy()

    column_name = ['Ag', 'Cu', 'Zn', 'Sn']
    print("Reconstructed Data (samples 90-93):")
    print(pd.DataFrame(recon_x.round(4), columns=column_name).loc[90:93])