import numpy as np

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU
from keras.layers import Add
from keras.utils import np_utils, Sequence
from keras import backend as K
from keras.callbacks import TensorBoard, Callback
from helpers import epoch_augmentation


class BatchStreamer(object):
    """
    Generates batches according to the given functions
    """
    @staticmethod
    def monte_carlo_batch(image_set, label_set, batch_size, context_size, patch_size):
        """
        Random Monte-Carlo sampling which generates a single batch to return of (images, labels)
        :param image_set: Set of images on which to sample
        :param label_set: Set of labels on which to sample
        :param batch_size: Size of batches
        :param context_size: Size of context to apply to each individual image
        :param patch_size: Size of patch on which to predict
        :return: Tuple of (images, labels) in NCHW format (Theano or TF) with N = batch_size.
        """
        image_batch = np.empty((batch_size, context_size, context_size, 3))
        label_batch = np.empty((batch_size, 2))

        for i in range(batch_size):
            # Select a random image
            idx = np.random.choice(image_set.shape[0])
            shape = image_set[idx].shape

            # Sample a random window from the image
            center = np.random.randint(context_size // 2, shape[0] - context_size // 2, 2)
            sub_image = image_set[idx][center[0] - context_size // 2:center[0] + context_size // 2,
                        center[1] - context_size // 2:center[1] + context_size // 2]
            gt_sub_image = label_set[idx][
                           center[0] - patch_size // 2:center[0] + patch_size // 2,
                           center[1] - patch_size // 2:center[1] + patch_size // 2]

            # Random flip
            if np.random.choice(2) == 0:
                # Flip vertically
                sub_image = np.flipud(sub_image)
            if np.random.choice(2) == 0:
                # Flip horizontally
                sub_image = np.fliplr(sub_image)

            # Random rotation in steps of 90°
            num_rot = np.random.choice(4)
            sub_image = np.rot90(sub_image, num_rot)

            # The label does not depend on the image rotation/flip (provided that the rotation is in steps of 90°)
            label = np.mean(gt_sub_image) > 0.25
            label = np_utils.to_categorical(label, 2)

            image_batch[i] = sub_image
            label_batch[i] = label

        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            image_batch = np.rollaxis(image_batch, 3, 1)

        return image_batch, label_batch

    @staticmethod
    def get_one_epoch(image_set, label_set, samples_per_epoch, batch_size, context_size, patch_size):
        """
        Return a tuple (images, labels) for a whole epoch worth of samplings.
        :param image_set: Set of images from which to generate the data from
        :param label_set: Set of labels from which to generate the data from
        :param samples_per_epoch: How many samples belong to an epoch
        :param batch_size: Size of each sample
        :param context_size: Size of full context (for images)
        :param patch_size: Size of individual patch on which to predict (mostly labels)
        :return: Tuple of (images, labels) as numpy-arrays in NCHW format (Theano or TF) with N == samples_per_epoch.
        """
        image_patches = []
        image_labels = []
        for i in range(samples_per_epoch):
            temp_patches, temp_labels = BatchStreamer.monte_carlo_batch(image_set, label_set, batch_size, context_size, patch_size)
            image_patches.append(temp_patches)
            image_labels.append(temp_labels)
        image_patches = np.reshape(
            np.asarray(image_patches), (samples_per_epoch * batch_size, context_size, context_size, 3)
        )
        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            image_patches = np.rollaxis(image_patches, 3, 1)
        image_labels = np.reshape(np.asarray(image_labels), (samples_per_epoch * batch_size, 2))
        return image_patches, image_labels


class ImageSequence(Sequence):
    """
    Custom sequencer used in the pipeline to return images in batches.
    """
    def __init__(self, x_set, y_set, x_aug, y_aug, batch_size, classes, context_size, patch_size, limit):
        self.x, self.y = x_set, y_set
        self.x_aug, self.y_aug = x_aug, y_aug
        self.batch_size = batch_size
        self.classes = classes
        self.context = context_size
        self.patch_size = patch_size
        self.padding = (context_size - patch_size) // 2
        self.limit = limit
        self.idx = 0

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        return BatchStreamer.monte_carlo_batch(self.x_aug, self.y_aug, self.batch_size, self.context, self.patch_size)
        # image_set = self.x_aug
        # label_set = self.y_aug
        # batch_size = self.batch_size
        # context_size = self.context
        # patch_size = self.patch_size
        #
        # image_batch = np.empty((batch_size, context_size, context_size, 3))
        # label_batch = np.empty((batch_size, 2))
        #
        # for i in range(batch_size):
        #     # Select a random image
        #     idx = np.random.choice(image_set.shape[0])
        #     shape = image_set[idx].shape
        #
        #     # Sample a random window from the image
        #     center = np.random.randint(context_size // 2, shape[0] - context_size // 2, 2)
        #     sub_image = image_set[idx][center[0] - context_size // 2:center[0] + context_size // 2,
        #                 center[1] - context_size // 2:center[1] + context_size // 2]
        #     gt_sub_image = label_set[idx][
        #                    center[0] - patch_size // 2:center[0] + patch_size // 2,
        #                    center[1] - patch_size // 2:center[1] + patch_size // 2]
        #
        #     # Random flip
        #     if np.random.choice(2) == 0:
        #         # Flip vertically
        #         sub_image = np.flipud(sub_image)
        #     if np.random.choice(2) == 0:
        #         # Flip horizontally
        #         sub_image = np.fliplr(sub_image)
        #
        #     # Random rotation in steps of 90°
        #     num_rot = np.random.choice(4)
        #     sub_image = np.rot90(sub_image, num_rot)
        #
        #     # The label does not depend on the image rotation/flip (provided that the rotation is in steps of 90°)
        #     label = np.mean(gt_sub_image) > 0.25
        #     label = np_utils.to_categorical(label, 2)
        #
        #     image_batch[i] = sub_image
        #     label_batch[i] = label
        #
        # if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
        #     image_batch = np.rollaxis(image_batch, 3, 1)
        #
        # return image_batch, label_batch

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx > self.limit:
            self.idx = 0
            raise StopIteration
        else:
            return self.__getitem__(self.idx - 1)

    def get_unmodified(self):
        return self.x, self.y, self.padding

    def overwrite_augmented(self, aug_img, aug_gt):
        self.x_aug = aug_img
        self.y_aug = aug_gt


class ImageShuffler(Callback):
    TRAINING_SET = None
    VALIDATION_SET = None

    def __init__(self, training_set, validation_set):
        super().__init__()
        self.TRAINING_SET = training_set
        self.VALIDATION_SET = validation_set

    def augment_images(self, images, groundtruths, padding):
        """
        Modify our images to have random modifications at each epoch.
        :return: Nothing
        """
        img_aug, gt_aug = [], []
        for idx in range(images.shape[0]):
            img_temp, gt_temp = epoch_augmentation(images[idx], groundtruths[idx], padding=padding)
            img_aug.append(img_temp)
            gt_aug.append(gt_temp)
        augmented_images = np.reshape(
            np.asarray(img_aug), (images.shape[0],
                                  images.shape[1] + 2 * padding,
                                  images.shape[2] + 2 * padding,
                                  images.shape[3])
        )
        augmented_groundtruth = np.reshape(
            np.asarray(gt_aug), (groundtruths.shape[0],
                                 groundtruths.shape[1] + 2 * padding,
                                 groundtruths.shape[2] + 2 * padding)
        )
        return augmented_images, augmented_groundtruth

    def on_epoch_begin(self, epoch, logs=None):
        img, gt, padding = self.TRAINING_SET.get_unmodified()
        img_aug, gt_aug = self.augment_images(img, gt, padding)
        self.TRAINING_SET.overwrite_augmented(img_aug, gt_aug)
        img, gt, padding = self.VALIDATION_SET.get_unmodified()
        img_aug, gt_aug = self.augment_images(img, gt, padding)
        self.VALIDATION_SET.overwrite_augmented(img_aug, gt_aug)


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''
    # https://github.com/keras-team/keras/issues/3358#issuecomment-312531958

    def __init__(self, val_data, batch_size, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.val_data = val_data      # Validation data of (patches, labels)
        self.batch_size = batch_size  # Size of single batch
        self.nb_steps = nb_steps      # Number of batches

    def on_epoch_end(self, epoch, logs):
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib = self.val_data[0][s * self.batch_size: (s + 1) * self.batch_size]
            tb = self.val_data[1][s * self.batch_size: (s + 1) * self.batch_size]
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.float32)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)


class ExtraMetrics(object):
    """
    Custom metrics for 2-class classification tasks.
    """

    # https://github.com/keras-team/keras/issues/5400#issuecomment-314747992
    @staticmethod
    def mcor(y_true, y_pred):
        # matthews_correlation
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        tp = K.sum(y_pos * y_pred_pos)
        tn = K.sum(y_neg * y_pred_neg)

        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)

        numerator = (tp * tn - fp * fn)
        denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return numerator / (denominator + K.epsilon())

    @staticmethod
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    @staticmethod
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def f1(y_true, y_pred):
        precision = ExtraMetrics.precision(y_true, y_pred)
        recall = ExtraMetrics.recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    @staticmethod
    def cil_error(y_true, y_pred):
        """Return the error rate based on dense predictions and 1-hot labels."""
        return 100.0 - (100 *
                K.cast(K.sum(K.cast(K.equal(K.argmax(y_pred, 1), K.argmax(y_true, 1)), 'int32')), 'float32') /
                K.cast(K.shape(y_pred)[0], 'float32')
        )


class BasicLayers(object):
    """
    Project API used for basic layers for functional Keras models.
    """
    DATA_FORMAT = None
    RELU_VERSION = None
    LEAKY_RELU_ALPHA = None

    def __init__(self, data_format='channels_first', relu_version=None, leaky_relu_alpha=0.01):
        self.DATA_FORMAT = data_format
        self.RELU_VERSION = relu_version
        self.LEAKY_RELU_ALPHA = leaky_relu_alpha


    def _conv2d(self, _input, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1), padding='same'):
        return Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=self.DATA_FORMAT,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None
        )(_input)

    def _batch_norm(self, _input, axis=1):
        # https://github.com/keras-team/keras/issues/1921#issuecomment-193837813
        # DO NOT TOUCH: We either use this after convolutions but use NCHW. As such we want to normalize on the
        #               features in the channels --> axis=1
        #               Else axis=1 is for our networks correct in that we use it AFTER flattening a 4D tensor into
        #               a 2D version. There the features are also on axis=1
        # In conclusion: You touch this---I will end you rightly *unscrews pommel*
        return BatchNormalization(
            axis=axis,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer='zeros',
            gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones',
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None
        )(_input)

    def _act_fun(self, _input, relu_version=RELU_VERSION):
        if relu_version == 'leaky':
            return LeakyReLU(alpha=self.LEAKY_RELU_ALPHA)(_input)
        elif relu_version == 'parametric':
            return PReLU(
                alpha_initializer='zeros',
                alpha_regularizer=None,
                alpha_constraint=None,
                shared_axes=None  # No sharing for channel-wise which is reportedly better
            )(_input)
        else:
            return ReLU()(_input)

    def _max_pool(self, _input, pool=(2, 2), strides=(2, 2), padding='same'):
        return MaxPooling2D(
            pool_size=pool,
            strides=strides,
            padding=padding,
            data_format=self.DATA_FORMAT
        )(_input)

    def _dense(self, _input, neurons):
        return Dense(
            units=neurons,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None
        )(_input)

    def _flatten(self, _input):
        return Flatten(data_format=self.DATA_FORMAT)(_input)

    def _spatialdropout(self, _input, rate=0.25):
        return SpatialDropout2D(rate=rate, data_format=self.DATA_FORMAT)(_input)

    def cbr(self, _input, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1), padding='same'):
        x = _input
        x = self._conv2d(x, filters, kernel_size, strides, dilation_rate, padding)
        x = self._batch_norm(x)
        x = self._act_fun(x)
        return x


class ResNetLayers(BasicLayers):
    """
    Helper class to generate quickly and consistently different forms of Residual Networks.
    """
    FULL_PREACTIVATION = False

    # ResNet constants
    FEATURES = [64, 128, 256, 512]
    REPETITIONS_SMALL  = [2, 2,  2, 2]
    REPETITIONS_NORMAL = [3, 4,  6, 3]
    REPETITIONS_LARGE  = [3, 4, 23, 3]
    REPETITIONS_EXTRA  = [3, 8, 36, 3]

    def __init__(self, data_format='channels_first', relu_version=None, leaky_relu_alpha=0.01, full_preactivation=False):
        super().__init__(data_format, relu_version, leaky_relu_alpha)
        self.FULL_PREACTIVATION = full_preactivation

    def _cbr(self, _input, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1), padding='same', no_act_fun=False):
        x = _input
        if not self.FULL_PREACTIVATION:
            x = self._conv2d(x, filters, kernel_size, strides, dilation_rate, padding)
        x = self._batch_norm(x)
        if not self.FULL_PREACTIVATION and no_act_fun:
            return x
        x = self._act_fun(x)
        if self.FULL_PREACTIVATION:
            x = self._conv2d(x, filters, kernel_size, strides, dilation_rate, padding)
        return x

    def stem(self, _input):
        x = _input
        x = self._conv2d(x, filters=64, kernel_size=(5, 5), strides=(2, 2))
        x = self._batch_norm(x)
        x = self._act_fun(x)
        x = self._max_pool(x, pool=(3, 3))
        return x

    def _vanilla_branch(self, _input, filters, strides, dilation=(1, 1)):
        x = _input
        x = self._cbr(x, filters=filters, kernel_size=(3, 3), strides=strides, dilation_rate=dilation)
        x = self._cbr(x, filters=filters, kernel_size=(3, 3), strides=(1, 1), dilation_rate=dilation, no_act_fun=True)
        return x

    def _bottleneck_branch(self, _input, filters, strides, dilation=(1, 1)):
        x = _input
        x = self._cbr(x, filters=filters, kernel_size=(1, 1), strides=strides, dilation_rate=dilation)
        x = self._cbr(x, filters=filters, kernel_size=(3, 3), strides=(1, 1), dilation_rate=dilation)
        x = self._cbr(x, filters=4 * filters, kernel_size=(1, 1), strides=(1, 1), dilation_rate=dilation, no_act_fun=True)
        return x

    def _short_branch(self, _input, filters, strides, dilation=(1, 1)):
        x = _input
        x = self._cbr(x, filters=filters, kernel_size=(3, 3), strides=strides, dilation_rate=dilation, no_act_fun=True)
        return x

    def _shortcut(self, _input, filters, strides=(1, 1), is_bottleneck=False):
        x = _input
        first_filters = filters
        if is_bottleneck:
            first_filters = 4 * filters
        x = self._cbr(x, first_filters, kernel_size=(1, 1), strides=strides, no_act_fun=True)
        return x

    def vanilla(self, _input, filters, is_first=False):
        if is_first:
            if filters == 64:
                strides = 1
            else:
                strides = 2
            shortcut = self._shortcut(_input, filters, strides)
            residual = self._vanilla_branch(_input, filters, strides)
        else:
            shortcut = _input
            residual = self._vanilla_branch(_input, filters, strides=(1, 1))
        res = Add()([shortcut, residual])
        return res

    def bottleneck(self, _input, filters, is_first=False):
        if is_first:
            if filters == 64:
                strides = 1
            else:
                strides = 2
            shortcut = self._shortcut(_input, filters, strides, True)
            residual = self._bottleneck_branch(_input, filters, strides)
        else:
            shortcut = _input
            residual = self._bottleneck_branch(_input, filters, strides=(1, 1))
        res = Add()([shortcut, residual])
        return res

    def short(self, _input, filters, is_first=False):
        if is_first:
            if filters == 64:
                strides = 1
            else:
                strides = 2
            shortcut = self._shortcut(_input, filters, strides)
            residual = self._short_branch(_input, filters, strides)
        else:
            shortcut = _input
            residual = self._short_branch(_input, filters, strides=(1, 1))
        res = Add()([shortcut, residual])
        return res