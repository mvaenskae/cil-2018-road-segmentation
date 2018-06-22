# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import numpy as np
from imgaug import augmenters as iaa
import re
import os


def load_image(path):
    """
    Load and return the path passed as an image.
    :param path: Path which should be an image
    :return: Read image
    """
    return mpimg.imread(path)


def pad_image(image, padding):
    """
    Pad an image's canvas by the amount of padding while filling the padded area with a reflection of the data.
    :param image: Image to pad in either [H,W] or [H,W,3]
    :param padding: Amount of padding to add to the image
    :return: Padded image, padding uses reflection along border
    """
    if len(image.shape) < 3:  # Grayscale image
        # Greyscale image (ground truth)
        image = np.lib.pad(image, ((padding, padding), (padding, padding)), 'reflect')
    elif len(image.shape) == 3:  # RGB image
        image = np.lib.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    else:
        assert False, "Method cannot pad 4D images"
    return image


def crop_gt(im, w, h, stride):
    """
    Crop a 2D image into patches of size [w, h] using a fixed stride.
    :param im: Image to crop ground truth labels from
    :param w: Width of the crops
    :param h: Height of the crops
    :param stride: Stride of the crops
    :return: List of patches in row-major ordering from image
    """
    assert len(im.shape) == 2, 'Expected grayscale image.'
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    for i in range(0, imgheight, stride):
        for j in range(0, imgwidth, stride):
            im_patch = im[j:j + w, i:i + h]
            list_patches.append(im_patch)
    return list_patches


def img_crop(im, w, h, stride, padding):
    """
    Crop a 2D image into patches of size [w, h] using a fixed stride after padding and mirroring the image at boundary.
    :param im: Image to crop ground truth labels from
    :param w: Width of the crops
    :param h: Height of the crops
    :param stride: Stride of the crops
    :return: List of patches in row-major ordering from image
    """
    assert len(im.shape) == 3, 'Expected RGB image.'
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    im = np.lib.pad(im, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    for i in range(padding, imgheight + padding, stride):
        for j in range(padding, imgwidth + padding, stride):
            im_patch = im[j - padding:j + w + padding, i - padding:i + h + padding, :]
            list_patches.append(im_patch)
    return list_patches


def create_patches(X, patch_size, stride, padding):
    img_patches = np.asarray([img_crop(X[i], patch_size, patch_size, stride, padding) for i in range(X.shape[0])])
    img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3], img_patches.shape[4])
    return img_patches


def create_patches_gt(X, patch_size, stride):
    img_patches = np.asarray([crop_gt(X[i], patch_size, patch_size, stride) for i in range(X.shape[0])])
    img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3])
    return img_patches


def group_patches(patches, num_images):
    return patches.reshape(num_images, -1)


def mask_to_submission(model, image_filename):
    """
    Generate prediction on image_filename using the model
    :param model: Model used for predictions
    :param image_filename: Image to open and predict on
    :return: Nothing
    """
    """ Reads a single image and outputs the strings that should go into the submission file. """
    img_number = int(re.search(r"\d+", image_filename).group(0))
    image = mpimg.imread(image_filename)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    prediction = model.classify(image)
    prediction = prediction.reshape(-1)
    patch_size = 16
    iter = 0
    print("Processing " + image_filename)
    for j in range(0, image.shape[2], patch_size):
        for i in range(0, image.shape[1], patch_size):
            label = int(prediction[iter])
            iter += 1
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def generate_submission(model, submission_filename, *image_filenames):
    """ Generate a .csv containing the classification of the test set. """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission(model, fn))


def read_images_plus_labels():
    """
    Read images and ground truth maps from system and return tuple to them
    :return: Returns tuple of images and ground truth maps
    """
    root_dir = os.path.join("data", "training")
    image_dir = os.path.join(root_dir, "images")
    gt_dir = os.path.join(root_dir, "groundtruth")
    files = os.listdir(image_dir)
    images_np = np.asarray([load_image(os.path.join(image_dir, file)) for file in files])
    ground_truth_np = np.asarray([load_image(os.path.join(gt_dir, file)) for file in files])
    return images_np, ground_truth_np


def split_dataset(images, gt_labels, seed):
    """
    Generate a split of 15 images for training and validation (plus labels)
    :param images: Array of images
    :param gt_labels: Array of ground truth labels
    :param seed: Seed for repeatability
    :return: 4-tuple of [img_train, gt_train, img_validate, gt_validate]
    """
    validate_count = 15
    np.random.seed(seed)
    image_count = len(images)
    train_count = image_count - validate_count
    index_array = list(range(image_count))
    permuted_indexes = np.random.permutation(index_array)
    validate_indexes = permuted_indexes[:validate_count]
    train_indexes = permuted_indexes[validate_count:]
    assert len(train_indexes) == train_count, "Index calculation errors on generating datasplit"

    img_train = []
    gt_train = []
    for idx in train_indexes:
        img_train.append(images[idx])
        gt_train.append(gt_labels[idx])

    img_validate = []
    gt_validate = []
    for idx in validate_indexes:
        img_validate.append(images[idx])
        gt_validate.append(gt_labels[idx])

    return np.asarray(img_train), np.asarray(gt_train), np.asarray(img_validate), np.asarray(gt_validate)


def predict_on_images(path, model, submission_filename):
    """
    Do predictions on images saved in the given path.
    :param path: Path to images
    :return: Nothing
    """
    image_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        image_filenames.extend(filenames)
        break

    image_paths = []
    for file in image_filenames:
        image_paths.append(os.path.join(path, file))

    image_paths = sorted(image_paths)
    generate_submission(model, submission_filename, *image_paths)


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def epoch_augmentation(__data, __ground_truth, padding):
    MAX = 2*padding
    assert (__data.shape != __ground_truth.shape), "Incorrect dimensions for data and labels"
    assert (MAX > 0), "Augmentation would reduce images, is this really what you want?"

    offset_x, offset_y = np.random.randint(0, MAX + 1, 2)
    padding = iaa.Pad(
        px=(offset_y, offset_x, MAX - offset_y, MAX - offset_x),
        pad_mode=["reflect"],
        keep_size=False
    )
    affine = iaa.Affine(
        rotate=(-180, 180),
        shear=(-5, 5),
        scale=(0.9, 1.1),
        mode=["reflect"]
    )
    augment_both = iaa.Sequential(
        [
            padding,                    # Pad the image to requested padding
            iaa.Sometimes(0.3, affine)  # Apply sometimes more interesting augmentations
        ],
        random_order=False
    ).to_deterministic()

    augment_image = iaa.Sequential(
        iaa.SomeOf((0, None), [                     # Run up to all operations
            iaa.ContrastNormalization((0.8, 1.2)),  # Contrast modifications
            iaa.Multiply((0.8, 1.2)),               # Brightness modifications
            iaa.Dropout(0.01),                      # Drop out single pixels
            iaa.SaltAndPepper(0.01)                 # Add salt-n-pepper noise
        ], random_order=True)                       # Randomize the order of operations
    ).to_deterministic()

    __data = img_float_to_uint8(__data)
    aug_image = augment_both.augment_image(__data)
    aug_ground_truth = augment_both.augment_image(__ground_truth)
    aug_image = augment_image.augment_image(aug_image)
    aug_image = aug_image / 255.0

    return aug_image, aug_ground_truth