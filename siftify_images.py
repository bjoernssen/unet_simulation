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
    input_images, target_masks = simulation.generate_random_data(192, 192, count=1)
    input_images_rgb = [x.astype(np.uint8) for x in input_images]
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
    # helper.plot_side_by_side([input_images_rgb, target_masks_rgb])
    # plt.show()

    sift = cv.xfeatures2d.SIFT_create(500)
    i = 0
    image = Image.open('Tumor_MRI/Yes/Image/TCGA_CS_4941_19960909_12.tif')
    img = np.array(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_mask = cv.cvtColor(target_masks_rgb[0], cv.COLOR_BGR2GRAY)
    channel = np.unique(gray_mask)
    kp, des = sift.detectAndCompute(img, None)
    points2f = cv.KeyPoint_convert(kp)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    plt.imshow(img)
    plt.show()
    # filename = 'sift_keypoints' + str(i) + '.jpg'

    # cv.imwrite(filename, img_kp)
    # img_segments = 1 + segmentation.slic(gray, compactness=1, n_segments=100)
    # superpixels = color.label2rgb(img_segments, gray, kind='avg')
    # plt.imshow(superpixels)
    # regions = regionprops(img_segments, intensity_image=rgb2gray(img))
    # for props in regions:
    #     cy, cx = props.centroid
    #     plt.plot(cx, cy, 'ro')
    # plt.show()

    img_segments2 = 1 + segmentation.slic(gray, compactness=0.1, n_segments=800)
    superpixels2 = color.label2rgb(img_segments2, gray, kind='avg')
    plt.imshow(img)
    regions2 = regionprops(img_segments2, intensity_image=rgb2gray(img))
    for props in regions2:
        cy, cx = props.centroid
        plt.plot(cx, cy, 'r.')
    plt.show()
    plt.imshow(superpixels2)
    plt.show()
    kp_pos, kp_val, y = keypoint_function.random_sim_kp(
        gray,
        thresh=10,
        n_kp=500
    )
    plt.imshow(img)
    for i in range(len(kp_pos)):
        plt.plot(int(kp_pos[i][1]), int(kp_pos[i][0]), 'r.')
    plt.show()

    img_kp = cv.drawKeypoints(gray, kp, img)
    plt.imshow(img_kp)
    plt.show()
    i = i + 1


