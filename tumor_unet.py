import os, sys
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from random import randrange
from utils.keypoint_function import draw_keypoints, random_keypoints

if __name__ == '__main__':
    gray_list = []
    n_kp = 100
    # for file in os.listdir('./Tumor'):
    filename = './Tumor/Y1.jpg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[gray < 40] = 0
    keypoint = 0
    kp_pos = []
    kp_value = []
    while keypoint < n_kp:
        rand_row = randrange(1200)
        rand_col = randrange(1200)
        if gray[rand_row, rand_col] != 0:
            kp_pos.append([rand_row, rand_col])
            kp_value.append(gray[rand_row, rand_col])
            keypoint += 1
    for kp in kp_pos:
        gray = cv2.circle(img, (kp[0], kp[1]), 3, color=(0,0,255), thickness=1, lineType=8, shift=0)
    cv2.imwrite('out.jpg', gray)
    gray_list.append(gray)
