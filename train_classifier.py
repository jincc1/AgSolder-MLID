import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
from utils.dataset import AttributeDataset
from matplotlib.pyplot import MultipleLocator
from config import get_config

if __name__ == '__main__':
    # Get configuration
    config = get_config()
    params = config.params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = config.root

    # Load data and latents from previous steps (assuming they were saved)
    all_data = pd.read_csv(config.data_path, header=0).iloc[:, 0:18].to_numpy()
    raw_x = all_data[:128, [0, 1, 2, 5]]
    raw_y = all_data[:128, -1].reshape(-1, 1)
    # Assuming latents are saved in a file, e.g., 'latents.npy'
    latents = np.load(os.path.join(root, 'latents.npy'))

    # --- Classifier for INVAR prediction ---
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

    params['cls_bs'] = 30
    params['cls_lr'] = 1e-4
    params['cls_epoch'] = 100
    params['num_fold'] = 8
    params['label_y'] = np.where(raw_y > 700, 1, 0)
    params['latents'] = latents

    cls = Classifier().to(device)
    opt = Adam(cls.parameters(), lr=params['cls_lr'], weight_decay=0.)

    def training_Cls(model, optimizer, params, device):
        label_y = params['label_y']
        latents = params['latents']
        cls_epoch = params['cls_epoch']

        kf = KFold(n_splits=params['num_fold'])
        train_acc = []
        test_acc = []

        k = 1
        for train, test in kf.split(latents):
            x_train, x_test, y_train, y_test = latents[train], latents[test], label_y[train], label_y[test]
            cls_dataset = AttributeDataset(x_train, y_train)
            cls_dataloader = DataLoader(cls_dataset, batch_size=params['cls_bs'], shuffle=True)
            cls_testDataset = AttributeDataset(x_test, y_test)
            cls_testDataloader = DataLoader(cls_testDataset, batch_size=cls_testDataset.__len__(), shuffle=False)

            for epoch in range(cls_epoch):
                t = time.time()
                total_loss = []
                total_acc = []
                model.train()

                for i, data in enumerate(cls_dataloader):
                    x = data[0].to(device)
                    y = data[1].to(device)
                    y_pred = model(x)
                    loss = F.binary_cross_entropy(y_pred, y)
                    total_acc.append(torch.sum(torch.where(y_pred >= 0.5, 1, 0) == y).detach().cpu().numpy())
                    total_loss.append(loss.item())

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    for test_data in cls_testDataloader:
                        x = test_data[0].to(device)
                        y = test_data[1].to(device)
                        y_pred = model(x)
                        accuracy = torch.sum(torch.where(y_pred >= 0.5, 1, 0) == y) / y_pred.size(0)
                        test_loss = F.binary_cross_entropy(y_pred, y)

                print(f'[{epoch+1:03}/{cls_epoch}] loss:{sum(total_loss)/len(total_loss):.3f} test_loss:{test_loss.item():.3f} acc:{sum(total_acc)/cls_dataset.__len__():.3f} test_acc:{accuracy:.3f} time:{time.time()-t:.3f}')

            print('[{}/{}] train_acc: {:.04f} || test_acc: {:.04f}'.format(k, params['num_fold'], sum(total_acc)/cls_dataset.__len__(), accuracy.item()))
            train_acc.append(sum(total_acc)/cls_dataset.__len__())
            test_acc.append(accuracy.item())
            k += 1
        print('train_acc: {:.04f} || test_acc: {:.04f}'.format(sum(train_acc)/len(train_acc), sum(test_acc)/len(test_acc)))
        plt.figure()
        sns.set_style()
        plt.xlabel('number of folds')
        plt.ylabel('loss')
        x = range(1, params['num_fold'] + 1)
        sns.set_style("darkgrid")
        x_major_locator = MultipleLocator(1)
        ax = plt.gca()
        plt.plot(x, train_acc)
        plt.plot(x, test_acc, linestyle=':', c='steelblue')
        plt.legend(["train_accuracy", "test_accuracy"])
        ax.xaxis.set_major_locator(x_major_locator)
        plt.savefig(os.path.join(root,'classifier_accuracy.png'), dpi=300)
        return train_acc, test_acc

    train_acc, test_acc = training_Cls(cls, opt, params, device)