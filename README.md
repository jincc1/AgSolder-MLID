# WAE-based Material Analysis

This project implements a Wasserstein Autoencoder (WAE) for analyzing material compositions, specifically focusing on Ag-based solders.

## Project Structure

AgSolder-MLID/
├── models/
│   ├── __init__.py
│   └── wae.py
├── utils/
│   ├── __init__.py
│   ├── dataset.py
│   ├── plotting.py
│   └── utils.py
├── train.py
├── evaluate_reconstruction.py
├── visualize_latent_space.py
├── gmm_analysis.py
├── train_classifier.py
├── mcmc_sampling.py
├── config.py
├── requirements.txt
└── README.md



-   **`models/`**: Contains the WAE model definition (`wae.py`).
-   **`utils/`**: Contains utility functions for dataset handling, plotting, and other helper functions.
-   **`train.py`**: Script for training the WAE model.
-   **`evaluate_reconstruction.py`**: Script to check the reconstruction quality of the WAE.
-   **`visualize_latent_space.py`**: Script to visualize and export the latent space.
-   **`gmm_analysis.py`**: Script for Gaussian Mixture Model analysis.
-   **`train_classifier.py`**: Script to train