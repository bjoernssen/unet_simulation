from random import randrange
import cv2
import numpy as np
import math


def random_keypoints(gray_image, threshold, n_kp):
    gray_image[gray_image < threshold] = 0
    keypoint = 0
    kp_pos = []
    kp_value = []
    while keypoint < n_kp:
        rand_row = randrange(gray_image.shape[0])
        rand_col = randrange(gray_image.shape[0])
        if gray_image[rand_row, rand_col] != 0:
            kp_pos.append([rand_row, rand_col])
            kp_value.append(gray_image[rand_row, rand_col])
            keypoint += 1
    return kp_pos, kp_value


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
    edges = np.empty((len(kp_pos), len(kp_pos)))
    for i in range(0, len(kp_pos)):
        edges[i, i] = 1
        for j in range(i + 1, len(kp_pos)):
            dist = math.sqrt((kp_pos[i][0] - kp_pos[j][0]) ** 2 + (kp_pos[i][1] - kp_pos[j][1]) ** 2)
            if dist == 0:
                edges[i, j] = 1
            elif dist < 20:
                edges[i, j] = 1 / dist
                edges[j, i] = 1 / dist

    return edges
