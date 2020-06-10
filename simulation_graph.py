from utils import simulation, helper
from utils.keypoint_function import random_keypoints, draw_keypoints, generate_edges
import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    input_images, target_masks = simulation.generate_random_data(192, 192, count=3)
    input_images_rgb = [x.astype(np.uint8) for x in input_images]
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
    helper.plot_side_by_side([input_images_rgb, target_masks_rgb])
    plt.show()
    n_kp = 100
    threshold = 35
    image = np.ascontiguousarray(input_images_rgb[0], dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp_pos, kp_val = random_keypoints(gray, threshold, n_kp)
    edges = generate_edges(kp_pos)
    draw_keypoints(image, gray, kp_pos, 'output.jpg')
    # helper.plot_img_array(gray, ncol=1)
    # helper.plot_side_by_side([image, gray])
    plt.show()
    x = 1
