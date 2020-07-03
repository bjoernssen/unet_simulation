import time
import torch
from torch_geometric.data import DataLoader

import numpy as np
import torch.nn as nn

from models.datasets import create_simulation_graph_set, create_sift_sim_set, create_binary_sim_set
from models.nets import GUNET
from mlflow import log_param, log_metric, log_artifact, start_run


if __name__ == '__main__':
    """ Define pre-requisits"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 30
    loss = nn.NLLLoss()
    tumor_directory = 'kaggle3m/'
    train_n = 20
    test_n = 2
    n_kp = 200
    threshold = 15

    """Define changeable parameters"""
    keypoint_functions = ['SIFT', 'KNN', 'Rand']
    lr_rates = [1e-2, 1e-3, 1e-4]
    depths = [3, 4, 5]
    hidden_channels = [100, 150, 200]
    pooling_ratios = [0.5, 0.5]

    """Building the data set"""
    # rand_train = create_simulation_graph_set(n_kp=n_kp, thresh=15, n_elem=train_n)
    # rand_test = create_simulation_graph_set(n_kp=n_kp, thresh=15, n_elem=test_n)
    #
    # rand_train_loader = DataLoader(rand_train, batch_size=10, shuffle=True)
    # rand_test_loader = DataLoader(rand_test, batch_size=10, shuffle=True)

    rand_bin_train = create_binary_sim_set(n_kp=n_kp, thresh=10, n_elem=train_n)
    rand_bin_test = create_binary_sim_set(n_kp=n_kp, thresh=10, n_elem=test_n)

    rand_bin_train_loader = DataLoader(rand_bin_train, batch_size=2, shuffle=True)
    rand_bin_test_loader = DataLoader(rand_bin_test, batch_size=2, shuffle=True)

    # sift_train = create_sift_sim_set(800, train_n)
    # sift_test = create_sift_sim_set(800, test_n)
    #
    # sift_train_loader = DataLoader(sift_train, batch_size=10, shuffle=True)
    # sift_test_loader = DataLoader(sift_test, batch_size=10, shuffle=True)

    lr = 1e-3
    """Generate models from parameters"""
    for deep in depths:
        for hidden in hidden_channels:
            model = GUNET(
                in_ch=3,
                hid_ch=400,
                depth=deep,
                out_ch=2,
                pool_ratios=pooling_ratios
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

            """Start the training session"""
            with start_run():
                log_param('hidden_channel', hidden)
                log_param('pooling_ratios', pooling_ratios)
                log_param('learning_rate', lr)
                log_param('depth', deep)
                for epoch in range(epochs):
                    train_loss = []
                    val_loss = []
                    running_train_los = []
                    start_time = time.time()

                    for data in rand_bin_train_loader:
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
                            for data in rand_bin_test_loader:
                                model.eval()
                                data = data.to(device)
                                pred = model(data).max(1)[1]
                                correct = float(pred.eq(data.y).sum().item())
                                acc = (correct / data.num_nodes)
                                running_val_los.append(acc)

                    epoch_train_acc = np.mean(running_train_los)
                    print('Train loss: {}'.format(epoch_train_acc))
                    train_loss.append(epoch_train_acc)
                    log_metric('train_loss', epoch_train_acc)

                    epoch_val_acc = np.mean(running_val_los)
                    print('test loss: {}'.format(epoch_val_acc))
                    val_loss.append(epoch_val_acc)
                    log_metric('test_loss', epoch_val_acc)

                    time_elapsed = time.time() - start_time
                    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                    log_metric('time', time_elapsed)
