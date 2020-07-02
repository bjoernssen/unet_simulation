from random import randrange
import torch
import cv2
import numpy as np
import math
import random
import sys


def random_sim_kp(gray_image, thresh, n_kp):
    keypoint = 0
    kp_pos = []
    kp_value = []
    kp_y = []
    while keypoint < n_kp:
        rand_row = random.choice(np.where(gray_image > thresh)[0])
        rand_col = random.choice(np.where(gray_image > thresh)[1])
        if gray_image[rand_row, rand_col] < thresh:
            continue
        kp_pos.append([rand_row, rand_col])
        kp_value.append(gray_image[rand_row, rand_col])
        kp_y.append(1)
        keypoint += 1
    keypoint_pos = torch.tensor(kp_pos)
    keypoint_val = torch.tensor(kp_value, dtype=torch.float32).view(n_kp, 1)
    y = torch.tensor(kp_y, dtype=torch.long)
    return keypoint_pos, keypoint_val, y


def random_keypoints(gray_image, color_image=None, threshold=int, n_kp=int):
    # gray_image[gray_image < threshold] = 0
    keypoint = 0
    kp_pos = []
    kp_value = []
    kp_y = []
    if gray_image.max() > 0:
        while keypoint < n_kp*0.3:
            rand_row = random.choice(np.where(gray_image == 255)[0])
            rand_col = random.choice(np.where(gray_image == 255)[1])
            kp_pos.append([rand_row, rand_col])
            kp_value.append(color_image[rand_row, rand_col])
            kp_y.append(1)
            keypoint += 1
    while keypoint < n_kp:
        rand_row = random.choice(np.where(gray_image == 0)[0])
        rand_col = random.choice(np.where(gray_image == 0)[1])
        kp_pos.append([rand_row, rand_col])
        kp_value.append(color_image[rand_row, rand_col])

        kp_y.append(0)
        keypoint += 1
    keypoint_pos = torch.tensor(kp_pos)
    keypoint_val = torch.tensor(kp_value, dtype=torch.float32)
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


def distanceIn2D(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def maxDistances3(point_array):
    edges_start = []
    edges_end = []
    for i in range(0, len(point_array)):
        first = second = third = -sys.maxsize
        first_index = second_index = third_index = 0
        for j in range(i, len(point_array)):
            dist = distanceIn2D(point_array[i, :], point_array[j, :])
            if dist > first:
                third = second
                second = first
                first = dist
                third_index = second_index
                second_index = first_index
                first_index = j

            elif dist > second:
                third = second
                second = dist
                third_index = second_index
                second_index = j

            elif dist > third:
                third = dist
                third_index = j
        edges_start.append(i)
        edges_start.append(i)
        edges_start.append(i)
        edges_end.append(first_index)
        edges_end.append(second_index)
        edges_end.append(third_index)
    return torch.tensor([edges_start, edges_end])



def distanceIn2D(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def maxDistances3(point_array):
    result = np.empty([len(point_array), len(point_array)])
    result_list = []
    for i in range(0, len(point_array)):
        first = second = third = -sys.maxsize
        for j in range(i, len(point_array)):
            dist = distanceIn2D(point_array[i, :], point_array[j, :])
            if dist > first:
                third = second
                second = first
                first = dist
                result[i, j] = dist
                result[j, i] = dist
                result_list.append([i,j])

            elif dist > second:
                third = second
                second = dist
                result[i, j] = dist
                result[j, i] = dist
                result_list.append([i,j])

            elif dist > third:
                third = dist
                result[i, j] = dist
                result[j, i] = dist
                result_list.append([i,j])

            else:
                result[i, j] = 0
    return result, result_list
