import copy
import time
from collections import defaultdict
import torch
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader

from models.datasets import SimDataset
from models.nets import ResNetUNet
from utils import simulation, helper
from utils.net_functions import calc_loss, print_metrics, reverse_transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchsummary import summary
import torch.nn.functional as F
from mlflow import log_metric, log_param, log_artifact


def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    """Create random images and their respective masks"""
    input_images, target_masks = simulation.generate_random_data(192, 192, count=3)
    input_images_rgb = [x.astype(np.uint8) for x in input_images]
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
    helper.plot_side_by_side([input_images_rgb, target_masks_rgb])
    plt.show()

    """Define sets to be used"""
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])
    train_set = SimDataset(10, transform=trans)
    val_set = SimDataset(2, transform=trans)
    image_datasets = {
        'train': train_set, 'val': val_set
    }
    batch_size = 25
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }
    inputs, masks = next(iter(dataloaders['train']))

    print(inputs.shape, masks.shape)

    plt.imshow(reverse_transform(inputs[3]))
    plt.show()

    """Define model and devices to be used"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = ResNetUNet(n_class=6)
    model = model.to(device)

    # check keras-like model summary using torchsummary
    summary(model, input_size=(3, 224, 224))

    """Define optimizer and train the model"""
    num_class = 6
    model = ResNetUNet(num_class).to(device)

    optimizer_ft = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=60)

    """Evaluate model"""
    model.eval()
    test_dataset = SimDataset(3, transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    # pred = model(inputs)
    #
    pred = F.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print(pred.shape)

    input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

    target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
    pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

    helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])
    plt.show()
