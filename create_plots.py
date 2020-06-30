import matplotlib.pyplot as plt
import numpy as np
import utils.helper
import utils.simulation

# Generate some random images
input_images, target_masks = utils.simulation.generate_random_data(192, 192, count=2)

for x in [input_images, target_masks]:
    print(x.shape)
    print(x.min(), x.max())

# Change channel-order and make 3 channels for matplot
input_images_rgb = [x.astype(np.uint8) for x in input_images]

# Map each channel (i.e. class) to each color
target_masks_rgb = [utils.helper.masks_to_colorimg(x) for x in target_masks]

# Left: Input image (black and white), Right: Target mask (6ch)
utils.helper.plot_side_by_side([input_images_rgb, target_masks_rgb])
plt.show()
