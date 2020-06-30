import os.path as osp
import random
import time

from sklearn.metrics import f1_score

from utils import helper, simulation, keypoint_function
from scipy.sparse import coo_matrix
import torch
from torch.nn import Dropout
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_adj
from torchvision import transforms
import numpy as np
from models.datasets import SimDataset
import cv2
import torch.nn as nn
from models.nets import GUNET
from models.datasets import create_tumor_set, create_sift_tumor_set, random_pixel_tumor_set
from mlflow import log_artifact, log_param, log_metric, start_run

if __name__ == '__main__':
    # dataset = create_tumor_set()
    # dataset1, dataset_alt = create_sift_tumor_set()

    epochs = 30
    dataset2 = random_pixel_tumor_set()

    train_loader = DataLoader(
        dataset=dataset2[:30],
        batch_size=5,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=dataset2[30:],
        batch_size=2,
        shuffle=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Net = GUNET(in_ch=3, hid_ch=200, out_ch=2, depth=3, pool_ratios=[0.5, 0.5]).to(device)
    optimizer = torch.optim.Adam(Net.parameters(), lr=0.01, weight_decay=0.001)

    train_loss = []
    val_loss = []
    loss = nn.NLLLoss()
    with start_run():
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            start_time = time.time()
            running_train_acc = []

            for data in train_loader:
                data = data.to(device)
                Net.train()
                optimizer.zero_grad()
                out = loss(Net(data), data.y)
                out.backward()
                optimizer.step()
                acc = float(Net(data).max(1)[1].eq(data.y).sum().item()) / data.num_nodes
                running_train_acc.append(acc)

            else:
                running_val_acc = []

                with torch.no_grad():
                    for data in test_loader:
                        Net.eval()
                        data = data.to(device)
                        pred = Net(data).max(1)[1]
                        correct = float(pred.eq(data.y).sum().item())
                        acc = correct / data.num_nodes
                        running_val_acc.append(acc)

            epoch_train_acc = np.mean(running_train_acc)
            log_metric('train_acc', epoch_train_acc)
            print('Train acc: {}'.format(epoch_train_acc))
            train_loss.append(epoch_train_acc)

            epoch_val_acc = np.mean(running_val_acc)
            print('Train acc: {}'.format(epoch_val_acc))
            val_loss.append(epoch_val_acc)

            time_elapsed = time.time() - start_time
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    Net.save_state_dict('gunet.py')
