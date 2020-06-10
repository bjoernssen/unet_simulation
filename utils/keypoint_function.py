from random import randrange
import torch
import cv2
import numpy as np
import math
import random

def random_keypoints(gray_image, threshold, n_kp):
    # gray_image[gray_image < threshold] = 0
    keypoint = 0
    kp_pos = []
    kp_value = []
    kp_y = []
    kp_mask = []
    while keypoint < n_kp/2:
        rand_row = randrange(gray_image.shape[0])
        rand_col = randrange(gray_image.shape[0])
        if gray_image[rand_row, rand_col] > threshold:
            kp_pos.append([rand_row, rand_col])
            kp_value.append(gray_image[rand_row, rand_col])
            kp_y.append(1)
            keypoint += 1
    while keypoint < n_kp:
        rand_row = randrange(gray_image.shape[0])
        rand_col = randrange(gray_image.shape[0])
        if gray_image[rand_row, rand_col] < threshold:
            kp_pos.append([rand_row, rand_col])
            kp_value.append(gray_image[rand_row, rand_col])
            kp_y.append(0)
            keypoint += 1
    keypoint_pos = torch.tensor(kp_pos)
    keypoint_val = torch.tensor(kp_value, dtype=torch.float32).view(100,1)
    y = torch.tensor(kp_y, dtype=torch.long)
    return keypoint_pos, keypoint_val, y


def draw_keypoints(image, gray, kp_pos, output_file):
    for kp in kp_pos:
        gray = cv2.circle(
            image,
            (kp[1], kp[0]),
            3,
            color=(0, 0, 255),
            thickness=1,
            lineType=8,
            shift=0
        )
    cv2.imwrite(output_file, gray)


def generate_edges(kp_pos):
    edges_start = []
    edges_end = []
    for i in range(0, len(kp_pos)):
        for j in range(i + 1, len(kp_pos)):
            dist = math.sqrt((kp_pos[i][0] - kp_pos[j][0]) ** 2 + (kp_pos[i][1] - kp_pos[j][1]) ** 2)
            if dist < 20:
                edges_start.append(i)
                edges_end.append(j)
    edges = torch.tensor([edges_start, edges_end])
    return edges


def generate_random_masks(n_kp):
    train_mask = [] 
    test_mask = []
    val_mask = []
    for i in range(0, n_kp):
        train_mask.append(bool(random.getrandbits(1)))
        test_mask.append(bool(random.getrandbits(1)))
        val_mask.append(bool(random.getrandbits(1)))
        i += 1
    train_mask = torch.tensor(train_mask)
    test_mask = torch.tensor(test_mask)
    val_mask = torch.tensor(val_mask)
    return train_mask, test_mask, val_mask
