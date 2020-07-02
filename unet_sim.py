import numpy as np
from utils import simulation, helper
import os, sys

if __name__ == '__main__':
    input_images, target_masks = simulation.generate_random_data(
        192,
        192,
        count=200
    )

