#!/usr/bin/env python3

import os
import sys
from PIL import Image
import math
import matplotlib.image as mpimg
import numpy as np

label_file = 'submission.csv'

h = 16
w = h
imgwidth = 608
imgheight = imgwidth
nc = 3

prediction_tags = [105, 106, 107, 108, 10, 115, 116, 11, 121, 122, 123, 124, 128, 129, 12, 130, 131, 136, 137, 138, 139, 140, 142, 143, 144, 145, 14, 151, 152, 153, 154, 155, 157, 159, 15, 161, 162, 168, 169, 170, 174, 175, 176, 177, 186, 187, 189, 190, 191, 192, 196, 200, 201, 202, 204, 205, 206, 207, 208, 211, 215, 216, 218, 219, 21, 220, 221, 222, 223, 23, 25, 26, 27, 29, 36, 40, 41, 49, 50, 51, 54, 61, 64, 65, 69, 76, 79, 7, 80, 8, 90, 92, 93, 9]

# Convert an array of binary labels to a uint8
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def reconstruct_from_labels(image_id):
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(label_file)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+w, imgwidth)
        ie = min(i+h, imgheight)
        if prediction == 0:
            adata = np.zeros((w,h))
        else:
            adata = np.ones((w,h))

        im[j:je, i:ie] = binary_to_uint8(adata)

    Image.fromarray(im).save('prediction_' + '%d' % image_id + '.png')

    return im

for i in prediction_tags:
    reconstruct_from_labels(i)
   
