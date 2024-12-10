import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.mixture import GaussianMixture
from models.wae import WAE
from utils.dataset import FeatureDataset
from utils.utils import get_latents
from utils.plotting import plot_gmm
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

    # Get latents
    latents = get_latents(model, dataset, device)

    # --- Gaussian Mixture Model ---
    gm = GaussianMixture(n_components=6, random_state=1, init_params='kmeans').fit(latents)
    print('Average negative log likelihood:', -1 * gm.score(latents))
    plot_gmm(gm, latents, raw_x, raw_y, latents)
    plt.savefig(os.path.join(root, 'gmm_plot.png'), bbox_inches='tight', pad_inches=0.01)

    # Elbow method for determining the optimal number of GMM components
    scores = []
    for i in range(1, 8):
        gm = GaussianMixture(n_components=i, random_state=1, init_params='kmeans').fit(latents)
        print('Average negative log likelihood:', -1 * gm.score(latents))
        scores.append(-1 * gm.score(latents))

    plt.figure()
    sns.set_style("darkgrid")
    plt.scatter(range(1, 8), scores, color='green')
    plt.plot(range(1, 8), scores)
    plt.xlabel('Number of Components')
    plt.ylabel('Average Negative Log-Likelihood')
    plt.title('Elbow Method for GMM')
    plt.savefig(os.path.join(root, 'elbow_plot.png'), format='png', dpi=300)
    plt.show()