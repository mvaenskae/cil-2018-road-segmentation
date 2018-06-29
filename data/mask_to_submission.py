#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import re

foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch

prediction_tags = [105, 106, 107, 108, 10, 115, 116, 11, 121, 122, 123, 124, 128, 129, 12, 130, 131, 136, 137, 138, 139, 140, 142, 143, 144, 145, 14, 151, 152, 153, 154, 155, 157, 159, 15, 161, 162, 168, 169, 170, 174, 175, 176, 177, 186, 187, 189, 190, 191, 192, 196, 200, 201, 202, 204, 205, 206, 207, 208, 211, 215, 216, 218, 219, 21, 220, 221, 222, 223, 23, 25, 26, 27, 29, 36, 40, 41, 49, 50, 51, 54, 61, 64, 65, 69, 76, 79, 7, 80, 8, 90, 92, 93, 9]

# assign a label to a patch
def patch_to_label(patch):
    # patch = patch / 255.0
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            # if label == 1:
            #     patch = patch.astype('uint8')
            #     plt.imshow(patch * 255, cmap=plt.cm.gray)
            #     plt.show()
            #     print("[" + str(i) + ":" + str(i + patch_size) + ", " + str(j) + ":" + str(j + patch_size) + "]:")
            #     print(patch * 255)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            print("Converting " + fn)
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


if __name__ == '__main__':
    submission_filename = 'submission-fullcnn.csv'
    image_filenames = []
    for i in prediction_tags:
        image_filename = os.path.join('predictions_fullcnn', str('test_' + str(i) + '.png'))
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
