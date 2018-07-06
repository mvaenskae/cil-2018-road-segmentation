import numpy as np

from keras.layers import Dense, Flatten, Concatenate, Dropout
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, SpatialDropout2D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU, ELU
from keras.layers import Add, Lambda
from keras.utils import np_utils, Sequence
from keras import backend as K
from keras.callbacks import TensorBoard, Callback
from helpers import epoch_augmentation, epoch_augmentation_old, get_feature_maps


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
    def monte_carlo_maps(image_set, gt_set, batch_size, output_size):
        """
        Random Monte-Carlo sampling which generates a single batch to return of (images, gt_maps)
        :param image_set: Set of images on which to sample
        :param gt_set: Set of ground-truth maps on which to sample
        :param batch_size: Size of batches
        :param output_size: Size of requested output for images and ground-truth maps
        :return: Tuple of (images, gt_maps) in NCHW format (Theano or TF) with N = batch_size.
        """
        image_batch = np.empty((batch_size, output_size, output_size, 3))
        gt_batch = np.empty((batch_size, output_size, output_size, 2))

        for i in range(batch_size):
            # Select a random image
            idx = np.random.choice(image_set.shape[0])
            shape = image_set[idx].shape

            # Sample a random window from the image
            if shape[1] == output_size:  # Whole image requested, cannot apply Monte-Carlo sampling
                sub_image = image_set[idx]
                gt_sub_image = gt_set[idx]
            else:
                top_left = np.random.randint(shape[1] - output_size, size=2)
                sub_image = image_set[idx][top_left[0]:top_left[0] + output_size, top_left[1]:top_left[1] + output_size]
                gt_sub_image = gt_set[idx][top_left[0]:top_left[0] + output_size, top_left[1]:top_left[1] + output_size]

            # Random flip
            if np.random.choice(2) == 0:
                # Flip vertically
                sub_image = np.flipud(sub_image)
                gt_sub_image = np.flipud(gt_sub_image)
            if np.random.choice(2) == 0:
                # Flip horizontally
                sub_image = np.fliplr(sub_image)
                gt_sub_image = np.fliplr(gt_sub_image)

            # Random rotation in steps of 90°
            num_rot = np.random.choice(4)
            sub_image = np.rot90(sub_image, num_rot)
            gt_sub_image = np.rot90(gt_sub_image, num_rot)

            # Extract feature maps for classification
            gt_sub_image_labels = get_feature_maps(gt_sub_image)

            image_batch[i] = sub_image
            gt_batch[i] = gt_sub_image_labels

        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            image_batch = np.rollaxis(image_batch, 3, 1)
            gt_batch = np.rollaxis(gt_batch, 3, 1)

        return image_batch, gt_batch

    @staticmethod
    def get_one_epoch_batch(image_set, label_set, samples_per_epoch, batch_size, context_size, patch_size):
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


class AbstractImageSequence(Sequence):
    """
    Custom sequencer used in the pipeline to return images in batches by applying Monte Carlo sampling.
    """
    def __init__(self, x_set, y_set, batch_size, output_size, limit=None):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.output_size = output_size
        self.padding = (output_size - x_set.shape[2]) // 2
        self.x_aug, self.y_aug = None, None
        self.idx = 0
        if limit is not None:
            self.limit = limit
        else:
            self.limit = None

    def __len__(self):
        if self.limit is None:
            self.limit = int(np.ceil(len(self.x_aug) / float(self.batch_size)))
        return self.limit

    def __getitem__(self, idx):
        raise NotImplementedError('AbstractImageSequence::build_model is not yet implemented.')

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


class ImageSequenceLabels(AbstractImageSequence):
    """
    Custom sequencer used in the pipeline to return images in batches by applying Monte Carlo sampling.
    This class returns labels.
    """
    def __init__(self, x_set, y_set, batch_size, output_size, patch_size, limit=None):
        super().__init__(x_set, y_set, batch_size, output_size, limit)
        self.patch_size = patch_size
        self.padding = (output_size - patch_size) // 2

    def __getitem__(self, idx):
        assert (self.x_aug is not None), "Images are not augmented. The Sequencer doesn't work without augmented images."
        assert (self.y_aug is not None), "Ground truth images are not augmented according to requirements."
        return BatchStreamer.monte_carlo_batch(self.x_aug, self.y_aug, self.batch_size, self.output_size, self.patch_size)


class ImageSequenceHeatmaps(AbstractImageSequence):
    """
    Custom sequencer used in the pipeline to return images in batches by applying Monte Carlo sampling.
    This class returns heatmaps.
    """
    def __init__(self, x_set, y_set, batch_size, output_size, limit=None):
        super().__init__(x_set, y_set, batch_size, output_size, limit)
        self.padding = (608 - x_set.shape[2]) // 2

    def __getitem__(self, idx):
        assert (self.x_aug is not None), "Images are not augmented. The Sequencer doesn't work without augmented images."
        assert (self.y_aug is not None), "Ground truth images are not augmented according to requirements."
        return BatchStreamer.monte_carlo_maps(self.x_aug, self.y_aug, self.batch_size, self.output_size)


class ImageShuffler(Callback):
    """
    Callback to shuffle/augment unmodified images from ImageSequence and feed the modified versions back to it at the
    start of each epoch.
    """
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
        if epoch != 0:  # Try to fix some concurreny issue due to Keras' threading approach
            img, gt, padding = self.TRAINING_SET.get_unmodified()
            img_aug, gt_aug = self.augment_images(img, gt, padding)
            self.TRAINING_SET.overwrite_augmented(img_aug, gt_aug)
            img, gt, padding = self.VALIDATION_SET.get_unmodified()
            img_aug, gt_aug = self.augment_images(img, gt, padding)
            self.VALIDATION_SET.overwrite_augmented(img_aug, gt_aug)

    def on_train_begin(self, logs=None):
        self.on_epoch_begin(-1)


class ImageShufflerOld(ImageShuffler):
    def augment_images(self, images, groundtruths, padding):
        """
        Modify our images to have random modifications at each epoch.
        :return: Nothing
        """
        img_aug, gt_aug = [], []
        for idx in range(images.shape[0]):
            img_temp, gt_temp = epoch_augmentation_old(images[idx], groundtruths[idx], padding=padding)
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


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''
    # TODO: Validate for NCHW and if required fix it
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
        """
        Calculate Matthew's correlation. Not sure if this works with multiclass tensors correctly...
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Matthew's correlation
        """
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
    def cil_error(y_true, y_pred):
        """Return the error rate based on dense predictions and 1-hot labels."""
        return 100.0 - (100 *
                K.cast(K.sum(K.cast(K.equal(K.argmax(y_pred, 1), K.argmax(y_true, 1)), 'int32')), 'float32') /
                K.cast(K.shape(y_pred)[0], 'float32')
        )

    # recall found on: https://stackoverflow.com/a/41717938
    @staticmethod
    def recall_class(y_true, y_pred, class_id):
        """
        Recall for a specifc class. This is only verified for 2-class one-hot encoding to be working.
        :param y_true: True labels
        :param y_pred: Prediction labels
        :param class_id: Class we want to get recall on (located at [_, class] on the one-hot encoding)
        :return: Recall for specified class
        """
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        mask_positive_self = K.cast(K.equal(class_id_true, class_id), 'int32')
        true_positive_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * mask_positive_self
        class_rec = K.sum(true_positive_tensor) / K.maximum(K.sum(mask_positive_self), 1)
        return class_rec

    # based on recall some hacky arithmetics and knowledge of only binary classes
    @staticmethod
    def precision_class(y_true, y_pred, class_id):
        """
        Precision for a specifc class. This is only verified for 2-class one-hot encoding to be working.
        :param y_true: True labels
        :param y_pred: Prediction labels
        :param class_id: Class we want to get recall on (located at [_, class] on the one-hot encoding)
        :return: Precision for specified class
        """
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)

        # Generate maths for predictions for classes true positive labels
        mask_positive_self = K.cast(K.equal(class_id_true, class_id),
                                    'int32')  # This is own class true positive plus false negative
        true_positive_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * mask_positive_self

        # Generate maths for predictions for other classes false negatives (meaning our false positives)
        mask_positive_other = K.cast(K.equal(class_id_true, ((class_id - 1) * -1)),
                                     'int32')  # This is other class true positive plus false negative (so our true negatives and false positives)
        false_positive_tensor = K.cast(K.not_equal(class_id_true, K.argmin(y_pred, axis=-1)),
                                       'int32') * mask_positive_other

        # Now we have our true positives and false positives
        true_positive_count = K.sum(true_positive_tensor)
        false_positive_count = K.sum(false_positive_tensor)
        class_precision = true_positive_count / K.maximum((true_positive_count + false_positive_count), 1)
        return class_precision

    @staticmethod
    def road_f1(y_true, y_pred):
        """
        F1-score for class "road" (aka foreground).
        :param y_true: True labels
        :param y_pred: Prediction labels
        :return: F1 score for class "road"
        """
        class_id = 1  # Switched due to precision and recall working that way... sorry :(
        prec = ExtraMetrics.precision_class(y_true, y_pred, class_id)
        rec = ExtraMetrics.recall_class(y_true, y_pred, class_id)
        return 2 * ((prec * rec) / (prec + rec + K.epsilon()))

    @staticmethod
    def non_road_f1(y_true, y_pred):
        """
        F1-score for class "non-road" (aka background).
        :param y_true: True labels
        :param y_pred: Prediction labels
        :return: F1 score for class "non-road"
        """
        class_id = 0  # Switched due to precision and recall working that way... sorry :(
        prec = ExtraMetrics.precision_class(y_true, y_pred, class_id)
        rec = ExtraMetrics.recall_class(y_true, y_pred, class_id)
        return 2 * ((prec * rec) / (prec + rec + K.epsilon()))

    @staticmethod
    def macro_f1(y_true, y_pred):
        """
        Macro averaged F1 score for both classes.
        :param y_true: True labels
        :param y_pred: Prediction labels
        :return: Average F1 score of both classes
        """
        prec_road = ExtraMetrics.precision_class(y_true, y_pred, 1)
        rec_road = ExtraMetrics.recall_class(y_true, y_pred, 1)
        prec_bg = ExtraMetrics.precision_class(y_true, y_pred, 0)
        rec_bg = ExtraMetrics.recall_class(y_true, y_pred, 0)

        prec = (prec_road + prec_bg) / 2
        rec = (rec_road + rec_bg) / 2

        return 2 * ((prec * rec) / (prec + rec + K.epsilon()))

    @staticmethod
    def avg_f1(y_true, y_pred):
        """
        Simple unweighted average of two F1-scores.
        :param y_true: True labels
        :param y_pred: Prediction labels
        :return: Average F1 score of both classes
        """
        f1_road = ExtraMetrics.road_f1(y_true, y_pred)
        f1_non_road = ExtraMetrics.non_road_f1(y_true, y_pred)
        return (f1_road + f1_non_road) / 2

    @staticmethod
    def micro_f1(y_true, y_pred):
        """
        Micro averaged F1 score for both classes.
        :param y_true: True labels
        :param y_pred: Prediction labels
        :return: Average F1 score of both classes
        Micro-average of precision = (TP1+TP2)/(TP1+TP2+FP1+FP2) = (12+50)/(12+50+9+23) = 65.96
        Micro-average of recall = (TP1+TP2)/(TP1+TP2+FN1+FN2) = (12+50)/(12+50+3+9) = 83.78
        """
        # TODO: Adjust this for correct calculations, right now calculated macro f1
        prec_road = ExtraMetrics.precision_class(y_true, y_pred, 1)
        rec_road = ExtraMetrics.recall_class(y_true, y_pred, 1)
        prec_bg = ExtraMetrics.precision_class(y_true, y_pred, 0)
        rec_bg = ExtraMetrics.recall_class(y_true, y_pred, 0)

        prec = (prec_road + prec_bg) / 2
        rec = (rec_road + rec_bg) / 2

        return 2 * ((prec * rec) / (prec + rec + K.epsilon()))


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

    def _conv2dt(self, _input, filters, kernel_size, strides=(1, 1), padding='same'):
        return Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=self.DATA_FORMAT,
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

    def _act_fun(self, _input):
        if self.RELU_VERSION == 'leaky':
            return LeakyReLU(alpha=self.LEAKY_RELU_ALPHA)(_input)
        elif self.RELU_VERSION == 'parametric':
            return PReLU(
                alpha_initializer='zeros',
                alpha_regularizer=None,
                alpha_constraint=None,
                shared_axes=None  # No sharing for channel-wise which is reportedly better
            )(_input)
        elif self.RELU_VERSION == 'exponential':
            return ELU(alpha=1.0)(_input)
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

    def _dropout(self, _input, rate=0.25):
        return Dropout(rate=rate)(_input)

    def _spatialdropout(self, _input, rate=0.25):
        return SpatialDropout2D(rate=rate, data_format=self.DATA_FORMAT)(_input)

    def _dropout(self, _input, rate):
        return Dropout(rate, noise_shape=None, seed=None)(_input)

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

    def static_vars(**kwargs):
        def decorate(func):
            for k in kwargs:
                setattr(func, k, kwargs[k])
            return func

        return decorate

    def __init__(self, data_format='channels_first', relu_version=None, leaky_relu_alpha=0.01, full_preactivation=False):
        super().__init__(data_format, relu_version, leaky_relu_alpha)
        self.FULL_PREACTIVATION = full_preactivation

    def _cbr(self, _input, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1), padding='same', no_act_fun=False):
        x = _input
        if not self.FULL_PREACTIVATION:
            x = self._conv2d(x, filters, kernel_size, strides, dilation_rate, padding)
        x = self._batch_norm(x)
        # if not self.FULL_PREACTIVATION and no_act_fun:
        #     return x
        x = self._act_fun(x)
        if self.FULL_PREACTIVATION:
            x = self._conv2d(x, filters, kernel_size, strides, dilation_rate, padding)
        return x

    def _tcbr(self, _input, filters, kernel_size, strides=(1, 1), padding='same', no_act_fun=False):
        x = _input
        if not self.FULL_PREACTIVATION:
            x = self._conv2dt(x, filters, kernel_size, strides, padding)
        x = self._batch_norm(x)
        # if not self.FULL_PREACTIVATION and no_act_fun:
        #     return x
        x = self._act_fun(x)
        if self.FULL_PREACTIVATION:
            x = self._conv2dt(x, filters, kernel_size, strides, padding)
        return x

    def stem(self, _input):
        x = _input
        x = self._conv2d(x, filters=64, kernel_size=(5, 5), strides=(2, 2))
        x = self._batch_norm(x)
        x = self._act_fun(x)
        x = self._max_pool(x, pool=(3, 3))
        return x

    def _vanilla_branch(self, _input, filters, strides=(1, 1)):
        x = _input
        x = self._cbr(x, filters=filters, kernel_size=(3, 3), strides=strides)
        x = self._cbr(x, filters=filters, kernel_size=(3, 3), no_act_fun=True)
        return x

    def _bottleneck_branch(self, _input, filters, strides=(1, 1)):
        x = _input
        x = self._cbr(x, filters=filters, kernel_size=(1, 1))
        x = self._cbr(x, filters=filters, kernel_size=(3, 3), strides=strides)
        x = self._cbr(x, filters=filters*4, kernel_size=(1, 1), no_act_fun=True)
        return x

    def _short_branch(self, _input, filters, strides=(1, 1)):
        x = _input
        x = self._cbr(x, filters=filters, kernel_size=(3, 3), strides=strides, no_act_fun=True)
        return x

    def _shortcut(self, _input, filters, strides=(2, 2), is_bottleneck=False):
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
            residual = self._vanilla_branch(_input, filters)
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
            residual = self._bottleneck_branch(_input, filters)
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
            residual = self._short_branch(_input, filters)
        res = Add()([shortcut, residual])
        return res


class InceptionResNetLayer(BasicLayers):

    HALF_SIZE=False

    def __init__(self, data_format='channels_first', relu_version=None, leaky_relu_alpha=0.01, half_size=True):
        super().__init__(data_format=data_format, relu_version=relu_version, leaky_relu_alpha=leaky_relu_alpha)
        self.HALF_SIZE=half_size

    def stem(self, _input):
        x = _input
        kernel_large = 5 if self.HALF_SIZE else 7
        x = self.cbr(x, 32, kernel_size=(3, 3), strides=(2, 2), padding='valid')
        x = self.cbr(x, 32, kernel_size=(3, 3), padding='valid')
        x = self.cbr(x, 64, kernel_size=(3, 3))
        x1 = self.cbr(x, 96, kernel_size=(3, 3), strides=(2, 2), padding='valid')
        x2 = self._max_pool(x, pool=(3, 3), strides=(2, 2), padding='valid')
        x = Concatenate(axis=1)([x1, x2])
        x1 = self.cbr(x, 64, kernel_size=(1, 1))
        x1 = self.cbr(x1, 64, kernel_size=(kernel_large, 1))
        x1 = self.cbr(x1, 64, kernel_size=(1, kernel_large))
        x1 = self.cbr(x1, 96, kernel_size=(3, 3), padding='valid')
        x2 = self.cbr(x, 64, kernel_size=(1, 1))
        x2 = self.cbr(x2, 96, kernel_size=(3, 3), padding='valid')
        x = Concatenate(axis=1)([x1, x2])
        if not self.HALF_SIZE:
            x1 = self._max_pool(x, pool=(3, 3), strides=(2,2), padding='valid')
            x2 = self.cbr(x, 192, kernel_size=(3, 3), strides=(2, 2), padding='valid')
        else:
            x1 = self.cbr(x, 192, kernel_size=(3, 1))
            x1 = self.cbr(x1, 192, kernel_size=(1, 3))
            x2 = self.cbr(x, 192, kernel_size=(3, 3))
        x = Concatenate(axis=1)([x1, x2])
        return x

    def block16(self, _input):
        x = _input
        shortcut = x
        x1 = self.cbr(x, 32, kernel_size=(1, 1))
        x1 = self.cbr(x1, 48, kernel_size=(3, 3))
        x1 = self.cbr(x1, 64, kernel_size=(3, 3))
        x2 = self.cbr(x, 32, kernel_size=(1, 1))
        x2 = self.cbr(x2, 32, kernel_size=(3, 3))
        x3 = self.cbr(x, 32, kernel_size=(1, 1))
        x = Concatenate(axis=1)([x1, x2, x3])
        x = self._conv2d(x, 384, kernel_size=(1, 1))
        x = Lambda(lambda l: l * 0.17)(x)
        x = Add()([x, shortcut])
        x = self._act_fun(x)
        return x

    def block7(self, _input):
        x = _input
        x1 = self.cbr(x, 256, kernel_size=(1, 1))
        x1 = self.cbr(x1, 256, kernel_size=(3, 3))
        x1 = self.cbr(x1, 384, kernel_size=(3, 3), strides=(2, 2), padding='valid')
        x2 =  self.cbr(x, 384, kernel_size=(3, 3), strides=(2, 2), padding='valid')
        x3 = self._max_pool(x, pool=(3, 3), strides=(2, 2), padding='valid')
        x = Concatenate(axis=1)([x1, x2, x3])
        return x

    def block17(self, _input):
        x = _input
        shortcut = x
        kernel_large = 5 if self.HALF_SIZE else 7
        x1 = self.cbr(x, 128, kernel_size=(1, 1))
        x1 = self.cbr(x1, 160, kernel_size=(1, kernel_large))
        x1 = self.cbr(x1, 192, kernel_size=(kernel_large, 1))
        x2 = self.cbr(x, 192, kernel_size=(1, 1))
        x = Concatenate(axis=1)([x1, x2])
        x = self._conv2d(x, K.int_shape(_input)[1], kernel_size=(1, 1))
        x = Lambda(lambda l: l * 0.1)(x)
        x = Add()([x, shortcut])
        x = self._act_fun(x)
        return x

    def block18(self, _input):
        x = _input
        x1 = self.cbr(x, 256, kernel_size=(1, 1))
        x1 = self.cbr(x1, 288, kernel_size=(3, 3))
        x1 = self.cbr(x1, 320, kernel_size=(3, 3), strides=(2, 2), padding='valid')
        x2 = self.cbr(x, 256, kernel_size=(1, 1))
        x2 = self.cbr(x2, 288, kernel_size=(3, 3), strides=(2, 2), padding='valid')
        x3 = self.cbr(x, 256, kernel_size=(1, 1))
        x3 = self.cbr(x3, 384, kernel_size=(3, 3), strides=(2, 2), padding='valid')
        x4 = self._max_pool(x, pool=(3, 3), strides=(2, 2), padding='valid')
        x = Concatenate(axis=1)([x1, x2, x3, x4])
        return x

    def block19(self, _input):
        x = _input
        shortcut = x
        x1 = self.cbr(x, 192, kernel_size=(1, 1))
        x1 = self.cbr(x1, 224, kernel_size=(1, 3))
        x1 = self.cbr(x1, 256, kernel_size=(3, 1))
        x2 = self.cbr(x, 192, kernel_size=(1, 1))
        x = Concatenate(axis=1)([x1, x2])
        x = self._conv2d(x, K.int_shape(_input)[1], kernel_size=(1, 1))
        x = Lambda(lambda l: l * 0.2)(x)
        x = Add()([x, shortcut])
        x = self._act_fun(x)
        return x


class RedNetLayers(ResNetLayers):
    FULL_PREACTIVATION = False

    # RedNet constants
    FEATURES = [64, 128, 256, 512]
    FEATURES_UP = [512, 256, 128, 64]
    REPETITIONS_NORMAL = [3, 4, 6, 3]
    REPETITIONS_UP_NORMAL = [6, 4, 3, 3]

    def __init__(self, data_format='channels_first', relu_version=None, leaky_relu_alpha=0.01,
                 full_preactivation=False):
        super().__init__(data_format, relu_version, leaky_relu_alpha, full_preactivation=full_preactivation)

    def stem(self, _input):
        x = _input
        x = self._conv2d(x, filters=64, kernel_size=(5, 5), strides=(2, 2))
        x = self._batch_norm(x)
        x = self._act_fun(x)
        x1 = x
        x = self._max_pool(x, pool=(3, 3))
        return x, x1

    def last_block(self, _input):
        x = _input
        for i in range(3):
            x = self.residual_up(x, 64, is_last=False)
        return x

    def _vanilla_branch_down(self, _input, filters, strides=(1, 1)):
        x = _input
        x = self._cbr(x, filters=filters*2, kernel_size=(3, 3), strides=strides)
        x = self._cbr(x, filters=filters, kernel_size=(3, 3),  no_act_fun=True)
        return x

    def _bottleneck_branch_down(self, _input, filters, strides=(1, 1)):
        x = _input
        x = self._cbr(x, filters=filters//2, kernel_size=(1, 1))
        x = self._cbr(x, filters=filters, kernel_size=(3, 3), strides=strides)
        x = self._cbr(x, filters=filters*4, kernel_size=(1, 1), no_act_fun=True)
        return x

    def _branch_up(self, _input, filters):
        x = _input
        x = self._cbr(x, filters=filters, kernel_size=(3, 3))
        x = self._tcbr(x, filters=filters//2, kernel_size=(3, 3), strides=(2, 2), no_act_fun=True)
        return x

    def _branch_up_keep_filters(self, _input, filters):
        x = _input
        x = self._cbr(x, filters=filters, kernel_size=(3, 3))
        x = self._tcbr(x, filters=filters, kernel_size=(3, 3), strides=(2, 2), no_act_fun=True)
        return x

    def _shortcut_up(self, _input, filters):
        x = _input
        x = self._tcbr(x, filters=filters//2, kernel_size=(2, 2), strides=(2, 2), no_act_fun=True)
        return x

    def _shortcut_up_keep_filters(self, _input, filters):
        x = _input
        x = self._tcbr(x, filters=filters, kernel_size=(2, 2), strides=(2, 2), no_act_fun=True)
        return x

    def vanilla_down(self, _input, filters, is_first=False):
        if is_first:
            if filters == 64:
                strides = 1
            else:
                strides = 2
            shortcut = self._shortcut(_input, filters, strides)
            residual = self._vanilla_branch_down(_input, filters, strides)
        else:
            shortcut = _input
            residual = self._vanilla_branch_down(_input, filters)
        res = Add()([shortcut, residual])
        return res

    def bottleneck_down(self, _input, filters, is_first=False):
        if is_first:
            if filters == 64:
                strides = 1
            else:
                strides = 2
            shortcut = self._shortcut(_input, filters, strides, True)
            residual = self._bottleneck_branch_down(_input, filters, strides)
        else:
            shortcut = _input
            residual = self._bottleneck_branch_down(_input, filters)
        res = Add()([shortcut, residual])
        return res

    def residual_up(self, _input, filters, is_last=False):
        if is_last:
            if filters == 64:
                filters = 2*filters
            shortcut = self._shortcut_up(_input, filters)
            residual = self._branch_up(_input, filters)
        else:
            shortcut = _input
            residual = self._vanilla_branch(_input, filters)
        res = Add()([shortcut, residual])
        return res

    def residual_up_keep_filters(self, _input, filters, is_last=False):
        if is_last:
            if filters == 64:
                filters = 2*filters
            shortcut = self._shortcut_up_keep_filters(_input, filters)
            residual = self._branch_up_keep_filters(_input, filters)
        else:
            shortcut = _input
            residual = self._vanilla_branch(_input, filters)
        res = Add()([shortcut, residual])
        return res


    def agent_layer(self, _input, filters):
        x = _input
        x = self._cbr(x, filters, kernel_size=(1, 1))
        return x

