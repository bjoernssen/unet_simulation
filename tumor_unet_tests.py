from os import listdir

from models.nets import UnetTumor
from utils.loss import BinaryCrossEntropy
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import torch.nn as nn
import torch.nn.functional as F


if __name__ == '__main__':
    tumor_directory = 'kaggle_3m/'
    loss_function = nn.BCELoss()
    lr = 1e-3
    tumor_set = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UnetTumor((3, 256, 256)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 30
    train_loss = []
    val_loss = []

    for patient in listdir(tumor_directory):
        for image_file in listdir(tumor_directory + patient):
            if 'mask' in image_file:
                continue

            img = np.array(
                    Image.open(
                        tumor_directory + patient + '/' + image_file
                    )
                )

            mask = image_file.replace('.tif', '_mask.tif')
            msk = np.array(
                    Image.open(
                        tumor_directory + patient + '/' + mask
                    )
                )
            img = torch.tensor(img / 255)
            msk = torch.tensor(msk / 255)

            img = img.view(3, 256, 256)
            msk = msk.view(1, 256, 256)
            tumor_set.append((img, msk))
        if len(tumor_set) > 200:
            break

    random.shuffle(tumor_set)
    train_set = round(len(tumor_set)*0.9)
    train_loader = DataLoader(
        dataset=tumor_set[:train_set],
        batch_size=10,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=tumor_set[train_set:],
        batch_size=10,
        shuffle=True
    )
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start_time = time.time()
        running_train_loss = []

        for image, mask in train_loader:
            image = image.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)
            mask_predicted = model.forward(image)
            loss = loss_function(torch.sigmoid(mask_predicted), mask)

            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())

        else:
            running_val_loss = []

            with torch.no_grad():
                for image, mask in test_loader:
                    image = image.to(device, dtype=torch.float)
                    mask = mask.to(device, dtype=torch.float)
                    pred_mask = model.forward(image)
                    loss = loss_function(torch.sigmoid(pred_mask), mask)
                    running_val_loss.append(loss.item())




        epoch_train_loss = np.mean(running_train_loss)
        print('Train loss: {}'.format(epoch_train_loss))
        train_loss.append(epoch_train_loss)

        epoch_val_loss = np.mean(running_val_loss)
        print('Validation loss: {}'.format(epoch_val_loss))
        val_loss.append(epoch_val_loss)

        time_elapsed = time.time() - start_time
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
