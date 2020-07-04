import time
import torch
from torch_geometric.data import DataLoader

import numpy as np
import torch.nn as nn

from models.datasets import random_tumor_set, sift_tumor_set, knn_tumor_set
from models.nets import GUNET
from mlflow import log_param, log_metric, log_artifact, start_run


if __name__ == '__main__':
    """ Define pre-requisits"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 30
    loss = nn.NLLLoss()
    tumor_directory = 'kaggle3m/'
    train_n = 10
    test_n = 2
    n_kp = 200
    threshold = 15

    """Define changeable parameters"""
    keypoint_functions = ['SIFT', 'KNN', 'Rand']
    lr_rates = [1e-2, 1e-3, 1e-4]
    depths = [3, 4, 5]
    hidden_channels = [100, 150, 200]
    pooling_ratios = [0.5, 0.5]

    dataset = sift_tumor_set(100, 200)
    train_loader = DataLoader(dataset[:2], batch_size=2, shuffle=True)
    test_loader = DataLoader(dataset[1:], batch_size=1, shuffle=True)
    lr = 1e-3
    model = GUNET(
        in_ch=3,
        hid_ch=200,
        depth=3,
        out_ch=2,
        pool_ratios=pooling_ratios
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

    """Start the training session"""
    with start_run():
        log_param('hidden_channel', 600)
        log_param('pooling_ratios', pooling_ratios)
        log_param('learning_rate', lr)
        log_param('depth', 3)
        for epoch in range(epochs):
            train_loss = []
            val_loss = []
            running_train_los = []
            start_time = time.time()

            for data in train_loader:
                data = data.to(device)
                model.train()
                optimizer.zero_grad()
                out = loss(model(data), data.y)
                out.backward()
                optimizer.step()
                acc = float(model(data).max(1)[1].eq(data.y).sum().item()) / data.num_nodes
                running_train_los.append(acc)
            else:
                running_val_los = []
                with torch.no_grad():
                    for data in test_loader:
                        model.eval()
                        data = data.to(device)
                        pred = model(data).max(1)[1]
                        correct = float(pred.eq(data.y).sum().item())
                        acc = (correct / data.num_nodes)
                        running_val_los.append(acc)

            epoch_train_acc = np.mean(running_train_los)
            print('Train acc: {}'.format(epoch_train_acc))
            train_loss.append(epoch_train_acc)
            log_metric('train_loss', epoch_train_acc)

            epoch_val_acc = np.mean(running_val_los)
            print('test acc: {}'.format(epoch_val_acc))
            val_loss.append(epoch_val_acc)
            log_metric('test_loss', epoch_val_acc)

            time_elapsed = time.time() - start_time
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            log_metric('time', time_elapsed)
