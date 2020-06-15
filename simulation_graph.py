import torch

from utils import simulation, helper, keypoint_function
from utils.keypoint_function import random_keypoints, draw_keypoints, generate_edges
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import join
from skimage import segmentation, color
from skimage.io import imread
from skimage.future import graph
from torch_geometric.data import Data
from skimage.measure import regionprops
from math import isnan


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])


if __name__ == '__main__':
    directory_file = 'MRI-selection/'
    directory_mask = 'MRI-selection-masks/'
    data_list = []

    for scan, mask in zip(listdir(directory_file), listdir(directory_mask)):
        img = Image.open(directory_file + scan)
        msk = Image.open(directory_mask + mask)

        scan_array = np.array(img)
        mask_array = np.array(msk)

        scan_segments = 1 + segmentation.slic(scan_array, compactness=1, n_segments=17)
        g = graph.rag_mean_color(scan_array, scan_segments)
        edges = []
        kp_value = []
        kp_pos = []
        mask = []

        for start in g.adj._atlas:
            for stop in list(g.adj._atlas[start].keys()):
                edges.append([start, stop])

        regions = regionprops(scan_segments, intensity_image=rgb2gray(scan_array))
        for props in regions:
            cy, cx = props.weighted_centroid
            if (isnan(cy)) or (isnan(cx)):
               cy, cx = props.centroid
            kp_pos.append([cy, cx])
            kp_value.append(scan_array[int(round(cy)), int(round(cx))])
            mask_value = mask_array[int(round(cy)), int(round(cx))]
            if mask_value > 0:
                mask.append(1)
            else:
                mask.append(0)
        keypoint_pos = torch.tensor(kp_pos)
        keypoint_val = torch.tensor(kp_value, dtype=torch.float32)
        y = torch.tensor(mask, dtype=torch.long)
        train_mask, test_mask, val_mask = keypoint_function.generate_random_masks(len(mask))

        data = Data(
            x=keypoint_val,
            edge_index=edges,
            pos=keypoint_pos,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask
        )
        data_list.append(data)


    img = Image.open('MRI-selection/raw/TCGA_CS_4941_19960909_11.tif')

    img.show()
    imarray = np.array(img)
    x=1
