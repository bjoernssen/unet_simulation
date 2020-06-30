import os.path as osp
import random

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
from models.datasets import create_tumor_set, create_sift_tumor_set, random_pixel_tumor_set


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pool_ratios = [.75, 0.5]
        self.unet = GraphUNet(3, 200, 2,
                              depth=3, pool_ratios=pool_ratios)

    def forward(self):
        edge_index, _ = dropout_adj(data.edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        x = F.dropout(data.x, p=0.92, training=self.training)
        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=1)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


if __name__ == '__main__':
    # dataset = create_tumor_set()
    # dataset1, dataset_alt = create_sift_tumor_set()
    dataset2 = random_pixel_tumor_set()
    train_data = dataset2[:15]
    test_data = dataset2[15:]
    dataset_shuffled = random.shuffle(dataset2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    best_val_acc = test_acc = 0

    for epoch in range(1, 301):
        for data in dataset2:
            train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
