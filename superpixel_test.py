from skimage import segmentation, color
from skimage.io import imread
from skimage.future import graph
from skimage.measure import regionprops
from matplotlib import pyplot as plt
from utils import simulation, helper
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])


if __name__ == '__main__':
    n_seg = 500
    compact = 1
    compact_list = [0.01, 0.1, 1, 10, 100]

    input_images, target_masks = simulation.generate_random_data(192, 192, count=3)
    input_images_rgb = [x.astype(np.uint8) for x in input_images]
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
    helper.plot_side_by_side([input_images_rgb, target_masks_rgb])
    plt.show()

    img = imread('ich.jpg')
    img_segments = 1 + segmentation.slic(img, compactness=compact, n_segments=n_seg)
    g = graph.rag_mean_color(img, img_segments)
    edges = []
    for start in g.adj._atlas:
        for stop in list(g.adj._atlas[start].keys()):
            edges.append([start, stop])
    regions = regionprops(img_segments, intensity_image=rgb2gray(img))
    superpixels = color.label2rgb(img_segments, img, kind='avg')
    centroid_colours = []
    for props in regions:
        cy, cx = props.weighted_centroid
        centroid_colours.append(img[int(round(cy)), int(round(cx))])
        plt.plot(cx, cy, 'ro')
    plt.imshow(superpixels)
    plt.show()
    for comp in compact_list:
        img_segments = segmentation.slic(img, compactness=comp, n_segments=n_seg)
        superpixels = color.label2rgb(img_segments, img, kind='avg')
        plt.imshow(superpixels)
        plt.show()
