import random

from scipy.sparse import coo_matrix
from torch.utils.data import Dataset
from skimage.io import imread
from skimage import segmentation, color
from skimage.io import imread
from skimage.future import graph
from skimage.measure import regionprops
from torch_geometric.data import InMemoryDataset
import torch
from os import listdir
from os.path import isfile
from utils import simulation, helper, keypoint_function
import numpy as np
import cv2
from torch_geometric.data import Data, DataLoader
from PIL import Image
from math import isnan
import cv2 as cv


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])


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


class TumorSet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TumorSet, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['MRI-selection/' + file for file in listdir('MRI-selection/') if isfile('MRI-selection/' + file)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        """initialise values for image processing"""
        data_list = []
        n_seg = 50
        compact = 1
        file_names = self.raw_file_names
        masks = ['MRI-selection-masks/' + file for file in listdir('MRI-selection-masks/')]
        for scan, mask in zip(file_names, masks):
            img = Image.open(scan)
            msk = Image.open(mask)

            scan_array = np.array(img)
            mask_array = np.array(msk)

            scan_segments = 1 + segmentation.slic(
                scan_array,
                compactness=1,
                n_segments=100
            )
            g = graph.rag_mean_color(scan_array, scan_segments)
            """Initialise values for data model construction"""
            edges = []
            kp_value = []
            kp_pos = []
            mask = []
            """Create edges in the graph"""
            for start in g.adj._atlas:
                for stop in list(g.adj._atlas[start].keys()):
                    edges.append([start, stop])
            regions = regionprops(
                scan_segments,
                intensity_image=rgb2gray(scan_array)
            )
            """Collect kp positions and values"""
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
            """Format values to fit into the data_model"""
            keypoint_pos = torch.tensor(kp_pos)
            keypoint_val = torch.tensor(kp_value, dtype=torch.float32)
            y = torch.tensor(mask, dtype=torch.long)
            train_mask, test_mask, val_mask = keypoint_function.generate_random_masks(
                len(mask)
            )
            """Create data object from image data"""
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
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_simulation_graph_set(n_kp, thresh, n_elem):
    input_images, target_masks = simulation.generate_random_data(192, 192, count=n_elem)
    input_images_rgb = [x.astype(np.uint8) for x in input_images]
    # target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
    data_list = []
    for i in range(0, n_elem):
        image = np.ascontiguousarray(input_images_rgb[i], dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp_pos, kp_val, y = keypoint_function.random_sim_kp(gray, thresh=thresh, n_kp=n_kp)
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


def create_tumor_set():
    directory_file = 'MRI-selection/raw/'
    directory_mask = 'MRI-selection-masks/'
    data_list = []

    for scan, mask in zip(listdir(directory_file), listdir(directory_mask)):
        img = Image.open(directory_file + scan)
        msk = Image.open(directory_mask + mask)

        scan_array = np.array(img)
        mask_array = np.array(msk)

        scan_segments = 1 + segmentation.slic(scan_array, compactness=1, n_segments=500)
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
    return data_list


def create_sift_tumor_set():
    sift = cv2.xfeatures2d.SIFT_create(1000)
    directory_file = 'Tumor_MRI/Image/'
    directory_mask = 'Tumor_MRI/Mask/'
    data_list = []
    data_alternativ = []
    dataset_size = 200
    i = 0

    for scan in listdir(directory_file):
        mask = scan.replace('.tif', '_mask.tif')
        kp_pos = []
        kp_val = []
        mask_list = []

        img = Image.open(directory_file + scan)
        msk = Image.open(directory_mask + mask)

        scan_array = np.array(img)
        mask_array = np.array(msk)

        kp, des = sift.detectAndCompute(scan_array, None)
        if len(kp) > 80:
            del kp[79:len(kp)-1]
        key_pos = cv2.KeyPoint_convert(kp)

        for key in key_pos:
            kp_val.append(
                scan_array[
                    int(round(key[0])),
                    int(round(key[1]))
                ]
            )
            mask_value = mask_array[
                int(round(key[0])),
                int(round(key[1]))
            ]
            if mask_value > 0:
                mask_list.append(1)
            else:
                mask_list.append(0)

        keypoint_pos = torch.tensor(key_pos)
        keypoint_val = torch.tensor(kp_val, dtype=torch.float32)
        y = torch.tensor(mask_list, dtype=torch.long)
        edge_list = keypoint_function.maxDistances3(key_pos)

        train_mask, test_mask, val_mask = keypoint_function.generate_random_masks(len(keypoint_pos))
        data = Data(
            x=keypoint_val,
            edge_index=edge_list,
            pos=keypoint_pos,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask
        )
        data_list.append(data)
        if i < dataset_size:
            if len(keypoint_pos) == 80:
                data_alternativ.append(data)
                i += 1
    return data_list, data_alternativ


def random_pixel_tumor_set():
    tumor_directory = 'Tumor_MRI/Yes/Image/'
    tumor_mask_directory = 'Tumor_MRI/Yes/Mask/'
    no_tumor_directory = 'Tumor_MRI/No/Image/'
    no_tumor_mask_directory = 'Tumor_MRI/No/Mask/'

    dataset_size = 35
    i = 0
    data_list = []

    for scan in listdir(tumor_directory):
        mask = scan.replace('.tif', '_mask.tif')
        kp_pos = []
        kp_val = []
        mask_list = []

        img = Image.open(tumor_directory + scan)
        msk = Image.open(tumor_mask_directory + mask)

        scan_array = np.array(img)
        mask_array = np.array(msk)
        keypoint_pos, keypoint_val, y = keypoint_function.random_keypoints(
            mask_array,
            color_image=scan_array,
            threshold=0,
            n_kp=200
        )
        edge_list = keypoint_function.maxDistances3(keypoint_pos)
        train_mask, test_mask, val_mask = keypoint_function.generate_random_masks(len(keypoint_pos))
        data = Data(
            x=keypoint_val,
            edge_index=edge_list,
            pos=keypoint_pos,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask
        )
        data_list.append(data)
        i += 1
        if i > dataset_size/2:
            break

    for scan in listdir(tumor_directory):
        mask = scan.replace('.tif', '_mask.tif')
        kp_pos = []
        kp_val = []
        mask_list = []

        img = Image.open(tumor_directory + scan)
        msk = Image.open(tumor_mask_directory + mask)

        scan_array = np.array(img)
        mask_array = np.array(msk)
        keypoint_pos, keypoint_val, y = keypoint_function.random_keypoints(
            mask_array,
            color_image=scan_array,
            threshold=0,
            n_kp=200
        )
        edge_list = keypoint_function.maxDistances3(keypoint_pos)
        train_mask, test_mask, val_mask = keypoint_function.generate_random_masks(len(keypoint_pos))
        data = Data(
            x=keypoint_val,
            edge_index=edge_list,
            pos=keypoint_pos,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask
        )
        data_list.append(data)
        i += 1
        if i > dataset_size:
            break
    return data_list


def create_sift_sim_set(n_kp, n_elem):
    input_images, target_masks = simulation.generate_random_data(192, 192, count=n_elem)
    input_images_rgb = [x.astype(np.uint8) for x in input_images]
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
    sift = cv.xfeatures2d.SIFT_create(n_kp)
    data_list = []
    for img, i in zip(input_images_rgb, range(target_masks.shape[0])):
        msk = target_masks[i,:,:,:]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp_values = []
        kp_y = []
        kp, des = sift.detectAndCompute(img, None)
        if len(kp) < 25:
            continue
        kp_sampled = random.choices(kp, k=25)
        points2f = cv.KeyPoint_convert(kp_sampled)
        distances, edges = keypoint_function.maxDistances3(points2f)
        channel_list = []
        for keypoint in points2f:
            kp_values.append(
                gray[int(keypoint[0]), int(keypoint[1])]
            )
            for channel in range(6):
                if msk[channel, int(keypoint[0]), int(keypoint[1])] == 1:
                    kp_y.append(channel+1)
                    break
                elif channel == 5:
                    kp_y.append(0)
        data = Data(
            x=kp_values,
            edge_index=edges,
            pos=points2f,
            y=kp_y,
        )
        data_list.append(data)
    return data_list
