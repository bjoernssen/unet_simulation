from models.datasets import SimDataset
from utils import helper, simulation
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy
import sys
from math import sqrt


def distanceIn2D(p1, p2):
    return sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


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


if __name__ == '__main__':
    input_images, target_masks = simulation.generate_random_data(192, 192, count=5)
    input_images_rgb = [x.astype(np.uint8) for x in input_images]
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
    helper.plot_side_by_side([input_images_rgb, target_masks_rgb])
    plt.show()

    sift = cv.xfeatures2d.SIFT_create(100)
    i = 0
    for img in input_images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(img, None)
        points2f = cv.KeyPoint_convert(kp)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        img = cv.drawKeypoints(gray, kp, img)
        filename = 'sift_keypoints' + str(i) + '.jpg'
        cv.imwrite(filename, img)

        i = i + 1
