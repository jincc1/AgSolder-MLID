class Config:
    def __init__(self):
        self.data_path = 'data/Ag_based_solder.csv' # Changed the data file name
        self.root = './'  # Working directory
        self.seed = 1
        self.params = {
            'num_epochs': 200,
            'batch_size': 50,
            'lr': 1e-4,
            'weight_decay': 0.0,
            'sigma': 8,
            'mmd_lambda': 1e-4,
            'model_name': 'WAE_v1',
        }

def get_config():
    return Config()