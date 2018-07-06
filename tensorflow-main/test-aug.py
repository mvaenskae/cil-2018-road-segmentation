from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
import matplotlib.pyplot as plt
import util

ia.seed(33)
PIXEL_DEPTH = 255

train_data, train_label = util.load_train_img(tiling=False)


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


def augment_data_with_label(__data, __labels, visualize=False):
    """
    Augment single image including label.
    :param __data: Image of dims 400x400
    :param __label: Image of dims 400x400
    :param visualize: Visualize image augmentations
    :return: Tuple of images of size 608x608
    """
    MAX = 608 - __data.shape[0]
    assert (__data.shape != __labels.shape), "Incorrect dimensions for data and labels"
    assert (MAX > 0), "Augmentation would reduce images, is this really what you want?"

    offset_x, offset_y = np.random.randint(0, MAX + 1, 2)
    padding = iaa.Pad(
        px=(offset_y, offset_x, MAX - offset_y, MAX - offset_x),
        pad_mode=["symmetric", "reflect", "wrap"],
        keep_size=False
    )
    affine = iaa.Affine(
        rotate=(-90, 90),
        shear=(-5, 5),
        scale=(0.9, 1.1),
        mode=["symmetric", "reflect", "wrap"]
    )

    augment_both = iaa.Sequential(
        [
            padding,  # Pad the image to 608x608 dimensions
            iaa.Fliplr(0.5),  # Horizontal flipping
            iaa.Flipud(0.5),  # Vertical flipping
            iaa.Sometimes(0.3, affine)  # Apply sometimes more interesting augmentations
        ],
        random_order=True
    ).to_deterministic()

    augment_image = iaa.Sequential(
        iaa.SomeOf((0, None), [  # Run up to all operations
            iaa.ContrastNormalization((0.8, 1.2)),  # Contrast modifications
            iaa.Multiply((0.8, 1.2)),  # Brightness modifications
            iaa.Dropout(0.01),  # Drop out single pixels
            iaa.SaltAndPepper(0.01)  # Add salt-n-pepper noise
        ], random_order=True)  # Randomize the order of operations
    ).to_deterministic()

    __data = img_float_to_uint8(__data)
    aug_image = augment_both.augment_image(__data)
    aug_labels = augment_both.augment_image(__labels)
    aug_image = augment_image.augment_image(aug_image)

    if visualize:
        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        plt.imshow(__data)
        a.set_title('Input image')
        a = fig.add_subplot(2, 2, 2)
        plt.imshow(__labels)
        a.set_title('Input truth')
        a = fig.add_subplot(2, 2, 3)
        plt.imshow(aug_image)
        a.set_title('Augmented image')
        a = fig.add_subplot(2, 2, 4)
        plt.imshow(aug_labels)
        a.set_title('Augmented truth')
        plt.show()
    else:
        return aug_image, aug_labels


for i in range(len(train_data)):
    augment_data_with_label(train_data[i], train_label[i], visualize=True)
