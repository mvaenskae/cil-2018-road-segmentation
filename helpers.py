# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import numpy as np
from imgaug import augmenters as iaa
from PIL import Image
from scipy.signal import convolve2d
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


def generate_blocks(img, w):
    """
    Square crop a square image to parts of requested size (w, w)
    :param im: Image to crop
    :param w: Length of single side
    :return: List of patches in row-major ordering
    """
    list_blocks = []
    imgwidth = img.shape[0]
    assert imgwidth == img.shape[1], 'Expected square image'
    assert (imgwidth % w) == 0, 'Requested size does not evenly segment image'
    for i in range(0, imgwidth, w):
        for j in range(0, imgwidth, w):
            im_patch = img[i:i + w, j:j + w]
            list_blocks.append(im_patch)
    np_blocks = np.asarray(list_blocks)
    return np_blocks


def group_blocks(imgs, w):
    """
    Concatenate square blocks of subimages into a new square image of requested size (w, w)
    :param imgs: List of subimages in row-major ordering
    :return: Reconstructed image
    """
    assert (imgs.shape[0] != 0), 'Expected numpy array of subimages.'
    rows = []
    row_count = w // imgs.shape[1]
    col_count = w // imgs.shape[2]
    for j in range(row_count):
        row = np.hstack(imgs[j * col_count: (j+1) * col_count])
        rows.append(row)
    return np.vstack(rows)


def get_feature_maps(gt):
    """
    Generates feature maps for the given ground-truth map. First channel is background, second channel roads.
    :param gt: Ground-truth map
    :return: Feature maps for given ground-truth map as numpy array
    """
    feature_classification = np.ndarray(shape=(2, gt.shape[0], gt.shape[1]))
    feature_classification[0] = gt
    feature_classification[1] = ((gt - gt.max()) * -1)
    return np.moveaxis(feature_classification, 0, -1)


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


def post_process_prediction(prediction):
    #filter = np.ones((3,3)) / 9
    filterh  = np.zeros((3, 3))
    filterh[1,:] = 1/3

    filterv  = np.zeros((3, 3))
    filterv[:,1] = 1/3

    filterd  = np.identity(3) / 3
    filterd2 = np.fliplr(np.identity(3))/3

    filters = (filterh, filterv, filterd, filterd2)

    s = np.zeros((len(filters), prediction.shape[0], prediction.shape[1]))

    cntr = 0
    for f in filters:
        s[cntr, :, :] = convolve2d(prediction, f, mode='same', boundary='symm')
        cntr += 1

    res = s.max(0)

    return res


def get_prediction(model, image, post_process):
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    prediction = model.classify(image)

    if post_process:
        prediction = post_process_prediction(prediction)

    return prediction


def prediction_to_labels(prediction):
    labels = (prediction > 0.5)
    return labels


def mask_to_submission(model, image_filename, post_process):
    """
    Generate prediction on image_filename using the model
    :param model: Model used for predictions
    :param image_filename: Image to open and predict on
    :return: Nothing
    """
    """ Reads a single image and outputs the strings that should go into the submission file. """
    img_number = int(re.search(r"\d+", image_filename).group(0))
    image = mpimg.imread(image_filename)

    prediction = get_prediction(model, image, post_process)
    prediction = prediction_to_labels(prediction)
    prediction = prediction.reshape(-1)

    patch_size = 16
    iter = 0
    print("Processing " + image_filename)
    for j in range(0, image.shape[1], patch_size):
        for i in range(0, image.shape[0], patch_size):
            label = int(prediction[iter])
            iter += 1
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def generate_submission(model, path, submission_filename, post_process):
    """
    Generate a .csv containing the classification of the test set.
    :param path: path to input files
    """
    filenames = get_files_in_dir(path)
    image_full_names = prepend_path_to_filenames(path, filenames)

    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_full_names[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission(model, fn, post_process))


def get_prediction_heatmap(model, image_filename, post_process):
    """
    Generate prediction on image_filename using the model (FullCNN)
    :param model: Model used for predictions
    :param image_filename: Image to open and predict on
    :return: Nothing
    """
    image = mpimg.imread(image_filename)
    print("Predicting " + image_filename)
    prediction = model.classify(image)

    if post_process:
        prediction = post_process_prediction(prediction)

    return prediction


def generate_submission_heatmaps(model, path, submission_directory, post_process):
    """
    Generate pixel-perfect predictions by the model.
    :param path: path to input files
    """
    filenames = get_files_in_dir(path)
    image_full_names = prepend_path_to_filenames(path, filenames)

    if not os.path.isdir(submission_directory):
        os.mkdir(submission_directory)

    for i, fname in enumerate(filenames):
        prediction = Image.fromarray(get_prediction_heatmap(model, image_full_names[i], post_process))
        prediction.save(os.path.join(submission_directory, fname), 'PNG')



def prediction_mask(model, img, post_processing):
    """ Generate a label mask of the same size as the input image """
    input_image_shape = img.shape
    prediction = get_prediction(model, img, post_processing)
    prediction = prediction_to_labels(prediction)
    prediction = prediction.reshape(-1)

    overlay = np.empty((input_image_shape[0], input_image_shape[1]))

    patch_size = 16
    iter = 0
    for i in range(0, input_image_shape[1], patch_size):
        for j in range(0, input_image_shape[0], patch_size):
            label = int(prediction[iter])

            overlay[j:(j + patch_size), i:(i + patch_size)] = label

            iter += 1

    return overlay


def generate_overlay_images(model, path, post_processing):
    """
    Generate images with the prediction as overlay for easier visualization.
    :param path: input file path
    """
    filenames = get_files_in_dir(path)

    for fn in filenames[0:]:
        print("Creating overlay for " + fn)
        input = load_image(os.path.join(path, fn))
        mask = prediction_mask(model, input, post_processing)

        output_folder = 'predictions'

        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        color_mask = np.zeros((input.shape[0], input.shape[1], 3), dtype=np.uint8)
        color_mask[:, :, 0] = mask * 255

        input8 = img_float_to_uint8(input)

        background_img = Image.fromarray(input8, 'RGB').convert("RGBA")
        overlay_img = Image.fromarray(color_mask, 'RGB').convert("RGBA")

        blended = Image.blend(background_img, overlay_img, 0.2)
        blended.save(os.path.join(output_folder, fn))


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


def split_dataset(images, gt_labels):
    """
    Generate a split of 15 images for training and validation (plus labels)
    :param images: Array of images
    :param gt_labels: Array of ground truth labels
    :param seed: Seed for repeatability
    :return: 4-tuple of [img_train, gt_train, img_validate, gt_validate]
    """
    validate_count = 15
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

def get_files_in_dir(dir):
    image_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        image_filenames.extend(filenames)

    return image_filenames

def prepend_path_to_filenames(path, filenames):
    image_paths = []
    for file in filenames:
        image_paths.append(os.path.join(path, file))

    return image_paths


######################
# Image Augmentation #
######################

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

    image_concrete = iaa.Sequential(
        [
            iaa.Multiply((1.5, 1.7)),
            iaa.ContrastNormalization((1.5, 1.7))
        ],
        random_order=False
    ).to_deterministic()

    image_normal = iaa.Sequential(
        iaa.SomeOf((0, None), [                     # Run up to all operations
            iaa.ContrastNormalization((0.8, 1.2)),  # Contrast modifications
            iaa.Multiply((0.8, 1.2)),               # Brightness modifications
        ], random_order=True)                       # Randomize the order of operations
    ).to_deterministic()

    __data = img_float_to_uint8(__data)
    aug_image = augment_both.augment_image(__data)
    aug_ground_truth = augment_both.augment_image(__ground_truth)
    if np.random.sample() < 0.1:
        aug_image = image_concrete.augment_image(aug_image)
    else:
        aug_image = image_normal.augment_image(aug_image)
    aug_image = aug_image / 255.0

    return aug_image, aug_ground_truth
