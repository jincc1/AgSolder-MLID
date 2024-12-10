import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draws an ellipse with the given position and covariance.
    """
    ax = ax or plt.gca()

    # Convert covariance to axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

def plot_gmm(gm, X, raw_x, raw_y, latents, label=True, ax=None):
    """
    Plots the Gaussian Mixture Model.
    """
    fig, axs = plt.subplots(1, 1, figsize=(2, 2), dpi=200)
    ax = axs or plt.gca()
    labels = gm.fit(X).predict(X)
    if label:
        # Separate data based on content thresholds
        high_ag = (raw_x[:, 0] > 0.3)
        high_ag_latent = latents[high_ag]

        high_cu = (raw_x[:, 1] > 0.3)
        high_cu_latent = latents[high_cu]

        high_in = (raw_x[:, 2] < 0.05) & (raw_x[:, 0] > 0.05) & (raw_x[:, 1] > 0.05) & (raw_x[:, 3] > 0.05)
        high_in_latent = latents[high_in]

        # Scatter plots with different colors and markers
        scatter1 = axs.scatter(high_ag_latent[:, 0], high_ag_latent[:, 1], c='steelblue', alpha=.55, s=8, linewidths=0, label='Alloys w/o Ag')
        scatter2 = axs.scatter(high_cu_latent[:, 0], high_cu_latent[:, 1], c='firebrick', alpha=.65, s=14, linewidths=0, marker='^', label='Alloys w/o Cu')
        scatter3 = axs.scatter(high_in_latent[:, 0], high_in_latent[:, 1], c='indigo', alpha=.65, s=14, linewidths=0, marker='s', label='Alloys w/o In')
    else:
        ax.scatter(X[:, 0], X[:, 1], s=5, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gm.weights_.max()  # Calculate weight factor for normalization
    # Iterate through the means, covariances, and weights of the GMM to draw ellipses
    for pos, covar, w in zip(gm.means_, gm.covariances_, gm.weights_):
        draw_ellipse(tuple(pos), covar, alpha=0.75 * w * w_factor, facecolor='slategrey', zorder=-10)

def export_to_csv(dataset, color, filename):
    """
    Exports data to a CSV file.
    """
    data = {
        'X Coordinate': dataset[:, 0],
        'Y Coordinate': dataset[:, 1],
        'Color': color[:, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)