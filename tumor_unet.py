from utils import simulation, helper
import torch
from utils.keypoint_function import random_keypoints, draw_keypoints, generate_edges
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.nets import UNet
gc.collect()


def train_step(inputs, labels, optimizer, criterion, unet, width_out, height_out):
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = unet(inputs)
    # outputs.shape =(batch_size, n_classes, img_cols, img_rows)
    outputs = outputs.permute(0, 2, 3, 1)
    # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
    m = outputs.shape[0]
    outputs = outputs.resize(m*width_out*height_out, 2)
    labels = labels.resize(m*width_out*height_out)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate, criterion, optimizer, x_train, y_train, x_val, y_val, width_out, height_out):
    epoch_iter = np.ceil(x_train.shape[0] / batch_size).astype(int)
    t = trange(epochs, leave=True)
    for _ in t:
        total_loss = 0
        for i in range(epoch_iter):
            batch_train_x = torch.from_numpy(x_train[i * batch_size : (i + 1) * batch_size]).float()
            batch_train_y = torch.from_numpy(y_train[i * batch_size : (i + 1) * batch_size]).long()
            if use_gpu:
                batch_train_x = batch_train_x.cuda()
                batch_train_y = batch_train_y.cuda()
            batch_loss = train_step(batch_train_x , batch_train_y, optimizer, criterion, unet, width_out, height_out)
            total_loss += batch_loss
        if (_+1) % epoch_lapse == 0:
            val_loss = get_val_loss(x_val, y_val, width_out, height_out, unet)
            print("Total loss in epoch %f : %f and validation loss : %f" %(_+1, total_loss, val_loss))
    gc.collect()


if __name__ == '__main__':
    input_images, target_masks = simulation.generate_random_data(192, 192, count=3)
    width_in = 192
    height_in = 192
    width_out = 192
    height_out = 192
    batch_size = 3
    epochs = 1
    epoch_lapse = 50
    threshold = 0.5
    learning_rate = 0.01
    unet = UNet(in_channel=1, out_channel=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unet.parameters(), lr=0.01, momentum=0.99)
    train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate, criterion, optimizer, x_train, y_train, x_val, y_val, width_out, height_out)
