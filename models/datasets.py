from scipy.sparse import coo_matrix
from torch.utils.data import Dataset
import torch
from utils import simulation, helper, keypoint_function
import numpy as np
import cv2
from torch_geometric.data import Data, DataLoader


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]


def create_simulation_graph_set(n_kp, thresh, n_elem):
    input_images, target_masks = simulation.generate_random_data(192, 192, count=n_elem)
    input_images_rgb = [x.astype(np.uint8) for x in input_images]
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
    data_list = []
    for i in range(0, n_elem):
        image = np.ascontiguousarray(input_images_rgb[i], dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp_pos, kp_val, y = keypoint_function.random_keypoints(gray, thresh, n_kp)
        edges = keypoint_function.generate_edges(kp_pos)
        train_mask, test_mask, val_mask = keypoint_function.generate_random_masks(n_kp)
        # edges_coo = coo_matrix(edges)
        data = Data(
            x=kp_val,
            edge_index=edges,
            pos=kp_pos,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask
        )
        data_list.append(data)
    return data_list
