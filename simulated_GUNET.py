import os.path as osp

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
from models.datasets import create_simulation_graph_set


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pool_ratios = [.75, 0.5]
        self.unet = GraphUNet(1, 100, 2,
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

# def test(loader):
#     model.eval()
#     #
#     ys, preds = [], []
#     for data in loader:
#         ys.append(data.y)
#         with torch.no_grad():
#             out = model(data.x.to(device), data.edge_index.to(device))
#         preds.append((out > 0).float().cpu())
#
#     y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
#     return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


if __name__ == '__main__':
    n_kp = 100
    threshold = 35
    train_n = 10
    test_n = 2

    train_data = create_simulation_graph_set(n_kp, threshold, train_n)
    test_data = create_simulation_graph_set(n_kp, threshold, test_n)
    train_loader = DataLoader(train_data, batch_size=2)
    test_loader = DataLoader(test_data, batch_size=1)
    # data = train_data[0]


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    best_val_acc = test_acc = 0

    for epoch in range(1, 75):
        for data in train_data:
            train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
