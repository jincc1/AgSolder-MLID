import torch
import numpy as np
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
from models.wae import WAE
from config import get_config

def MCMC(gm, classifier, n_samples, device, sigma=0.1):
    """
    Performs Markov Chain Monte Carlo (MCMC) sampling.
    """
    sample_z = []

    z = gm.sample(1)[0]
    for i in range(n_samples):
        uniform_rand = np.random.uniform(size=1)
        z_next = np.random.multivariate_normal(z.squeeze(), sigma * np.eye(2)).reshape(1, -1)

        z_combined = np.concatenate((z, z_next), axis=0)
        scores = classifier(torch.Tensor(z_combined).to(device)).detach().cpu().numpy().squeeze()
        z_score, z_next_score = np.log(scores[0]), np.log(scores[1])
        z_prob, z_next_prob = (gm.score(z) + z_score), (gm.score(z_next) + z_next_score)
        acceptance = min(0, (z_next_prob - z_prob))

        if i == 0:
            sample_z.append(z.squeeze())

        if np.log(uniform_rand) < acceptance:
            sample_z.append(z_next.squeeze())
            z = z_next

    return np.stack(sample_z)

if __name__ == '__main__':
    # Get configuration
    config = get_config()
    params = config.params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = config.root

    # Load data and latents (assuming they were saved)
    all_data = pd.read_csv(config.data_path, header=0).iloc[:, 0:18].to_numpy()
    raw_x = all_data[:128, [0, 1, 2, 5]]
    # Assuming latents are saved in a file, e.g., 'latents.npy'
    latents = np.load(os.path.join(root, 'latents.npy'))

    # Load model
    model_dir = os.path.join(root, '{}/{}_{}.pth'.format(params['model_name'], params['model_name'], params['num_epochs']))
    model = WAE(raw_x.shape[1]).to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    # Load classifier (assuming it was saved)
    # Assuming the classifier is saved in a file, e.g., 'classifier.pth'
    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(2, 8),
                nn.Dropout(0.5),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.fc(x)

    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load(os.path.join(root, 'classifier.pth')))
    classifier.eval()

    # Fit GMM (you might want to load a pre-fitted GMM if available)
    gm = GaussianMixture(n_components=6, random_state=1, init_params='kmeans').fit(latents)

    # Sample using MCMC
    sample_z = MCMC(gm=gm, classifier=classifier, n_samples=50000, device=device, sigma=8)
    WAE_comps = model.decode(torch.Tensor(sample_z).to(device)).detach().cpu().numpy()
    print('Sample size:', sample_z.shape)

    # Save the generated compositions
    column_name = ['Ag', 'Cu', 'Zn', 'Sn']
    WAE_comps = pd.DataFrame(WAE_comps)
    WAE_comps.columns = column_name
    WAE_comps.to_csv(os.path.join(root,'comps_WAE.csv'), index=False)
    print(WAE_comps.head())