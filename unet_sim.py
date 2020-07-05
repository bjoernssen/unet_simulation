from skimage.measure import regionprops

from models.datasets import create_simulation_graph_set
from utils import helper, simulation, keypoint_function
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy
import sys
from math import sqrt
from skimage import segmentation, color
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])


if __name__ == '__main__':
    # input_images, target_masks = simulation.generate_random_data(192, 192, count=1)
    # input_images_rgb = [x.astype(np.uint8) for x in input_images]
    # target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
    # # helper.plot_side_by_side([input_images_rgb, target_masks_rgb])
    # # plt.show()
    # img = rgb2gray(input_images_rgb[0])
    # msk = rgb2gray(target_masks_rgb[0])
    image = Image.open('datasets/Tumor_MRI/Yes/Image/TCGA_CS_4941_19960909_12.tif')
    mask = Image.open('datasets/Tumor_MRI/Yes/Mask/TCGA_CS_4941_19960909_12_mask.tif')

    img = np.array(image)
    msk = np.array(mask)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    plt.imshow(img)
    plt.show()

    kp_pos, kp_val, y = keypoint_function.random_keypoints(
        msk,
        img,
        n_kp=500
    )
    plt.imshow(img)
    for i in range(len(kp_pos)):
        plt.plot(int(kp_pos[i][1]), int(kp_pos[i][0]), 'r.')
    plt.show()

    plt.imshow(img)
    for i in range(len(kp_pos)):
        if y[i] > 0:
            plt.plot(int(kp_pos[i][1]), int(kp_pos[i][0]), 'b.')
        elif y[i] == 0:
            plt.plot(int(kp_pos[i][1]), int(kp_pos[i][0]), 'r.')
    plt.show()



