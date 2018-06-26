# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam, SGD
from keras import losses
from keras import metrics
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from helpers import *
from keras_helpers import *

class CnnModel:
    """
    Base class for any CNN model.
    """
    def __init__(self, patch_size=16, context=64, data_format='channels_first', relu_version=None, leaky_relu_alpha=0.01):
        """ Construct a CNN classifier. """
        self.PATCH_SIZE = patch_size
        self.CONTEXT = context
        self.PADDING = (self.CONTEXT - self.PATCH_SIZE) // 2
        self.NB_CLASSES = 2
        self.BATCH_SIZE = 16
        self.DATA_FORMAT = data_format
        self.RELU_VERSION = relu_version
        self.LEAKY_RELU_ALPHA = leaky_relu_alpha
        self.model = None
        self.MODEL_NAME = 'Base'

        # Compatibility with Theano and Tensorflow ordering
        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            self.INPUT_SHAPE = (3, self.CONTEXT, self.CONTEXT)
        else:
            self.INPUT_SHAPE = (self.CONTEXT, self.CONTEXT, 3)

        self.build_model()
    
    def build_model(self):
        """
        This method is required to be overwritten by a child-class and supposed to generate the network.
        :return: Nothing
        """
        raise NotImplementedError('CnnModel::build_model is not yet implemented.')

    def train(self, epochs):
        """
        Train this model with the given dataset.
        """
        seed = 42
        np.random.seed(seed)
        nb_epoch = epochs

        # Read images from disk and generate training and validation set
        images, groundtruth = read_images_plus_labels()
        images_train, groundtruth_train, images_validate, groundtruth_validate = split_dataset(images, groundtruth, seed)

        # Print some extraneous metrics helpful to users of this template
        samples_per_image = images.shape[1] * images.shape[2] // (self.PATCH_SIZE * self.PATCH_SIZE)
        batches_train = (images_train.shape[0] * samples_per_image) // self.BATCH_SIZE
        batches_validate = (images_validate.shape[0] * samples_per_image) // self.BATCH_SIZE

        print('Dataset shape:', images.shape, '( Train:', images_train.shape[0], '| Validate:', images_validate.shape[0], ')')
        print('Samples per image:', samples_per_image, '( Trainsteps per epoch:', batches_train, '| Validatesteps per epoch:', batches_validate, ')')

        # Generators for image sequences which apply Monte Carlo sampling on them
        training_data = ImageSequence(images_train, groundtruth_train, self.BATCH_SIZE, self.NB_CLASSES, self.CONTEXT,
                                      self.PATCH_SIZE, batches_train)
        validation_data = ImageSequence(images_validate, groundtruth_validate, self.BATCH_SIZE, self.NB_CLASSES,
                                        self.CONTEXT, self.PATCH_SIZE, batches_validate)

        # Reduce learning rate iff validation average f1 score not improving for AdamOptimizer
        reduce_lr_on_plateau_adam = ReduceLROnPlateau(monitor='val_avg_f1',
                                                      factor=0.1,
                                                      patience=2,
                                                      verbose=1,
                                                      mode='max',
                                                      min_delta=1e-2,
                                                      cooldown=0,
                                                      min_lr=1e-7)

        # Stop training early iff validation average f1 score not improving for AdamOptimizer
        early_stopping_adam = EarlyStopping(monitor='val_avg_f1',
                                            min_delta=1e-3,
                                            patience=5,
                                            verbose=1,
                                            mode='max')

        # Reduce learning rate iff validation average f1 score not improving for SGD
        reduce_lr_on_plateau_sgd = ReduceLROnPlateau(monitor='val_avg_f1',
                                                     factor=0.5,
                                                     patience=5,
                                                     verbose=1,
                                                     mode='max',
                                                     min_delta=1e-4,
                                                     cooldown=0,
                                                     min_lr=1e-8)

        # Stop training early iff validation average f1 score not improving for AdamOptimizer
        early_stopping_sgd = EarlyStopping(monitor='val_avg_f1',
                                           min_delta=1e-4,
                                           patience=11,
                                           verbose=1,
                                           mode='max')

        # Enable Tensorboard logging and show the graph -- Other options not sensible when using Monte Carlo sampling
        tensorboard = TensorBoard(log_dir='./logs',
                                  histogram_freq=0,
                                  batch_size=self.BATCH_SIZE,
                                  write_graph=True,
                                  write_grads=False,
                                  write_images=False,
                                  embeddings_freq=0,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)

        # Hacky Tensorboard wrapper if a fixed validation set is given to model.generator_fit and this wrapper
        # tensorboard_hack = TensorBoardWrapper(validate_data,
        #                                       nb_steps=batches_validate,
        #                                       log_dir='./logs',
        #                                       histogram_freq=1,
        #                                       batch_size=batch_size,
        #                                       write_graph=True,
        #                                       write_grads=False,
        #                                       write_images=False)

        # Save the model's state on each epoch, given the epoch has better fitness
        filepath = "weights-" + self.MODEL_NAME + "-e{epoch:03d}-f1-{val_avg_f1:.4f}.hdf5"
        checkpointer = ModelCheckpoint(filepath=filepath,
                                       monitor='val_avg_f1',
                                       mode='max',
                                       verbose=1,
                                       save_best_only=True,
                                       period=1)

        # Shuffle/augment images at the start of each epoch
        image_shuffler = ImageShuffler(training_data, validation_data)

        def softmax_crossentropy_with_logits(y_true, y_pred):
            """
            Applies the loss function as found in the template not using logits (so use softmax layer at the end)
            :param y_true: Ground-truth values
            :param y_pred: Network predictions
            :return: Application of K.categorical_crossentropy()
            """
            return K.categorical_crossentropy(y_true, y_pred, from_logits=False, axis=1)

        # Define in a list what callbacks and metrics we want included
        model_callbacks_adam = [tensorboard, checkpointer, reduce_lr_on_plateau_adam, image_shuffler]
        model_callbacks_sgd = [tensorboard, checkpointer, reduce_lr_on_plateau_sgd, image_shuffler]
        model_metrics = [metrics.categorical_accuracy, ExtraMetrics.mcor, ExtraMetrics.cil_error, ExtraMetrics.road_f1,
                         ExtraMetrics.non_road_f1, ExtraMetrics.avg_f1]

        self.model.compile(loss=softmax_crossentropy_with_logits,
                           optimizer=Adam(lr=1e-4),
                           metrics=model_metrics)

        try:
            self.model.fit_generator(
                generator=training_data,
                steps_per_epoch=batches_train,
                epochs=nb_epoch,
                verbose=1,
                callbacks=model_callbacks_adam,
                validation_data=validation_data,
                validation_steps=batches_validate,
                shuffle=False,  #Not needed, our generator shuffles everything already
                use_multiprocessing=False)  # This requires a thread-safe generator which we don't have

            # TODO: Generate callback which makes this double-call to the network not required.
            self.model.compile(loss=softmax_crossentropy_with_logits,
                               optimizer=SGD(lr=1e-4, momentum=0.9, nesterov=False),
                               metrics=model_metrics)

            self.model.fit_generator(
                generator=training_data,
                steps_per_epoch=batches_train,
                epochs=nb_epoch,
                verbose=1,
                callbacks=model_callbacks_sgd,
                validation_data=validation_data,
                validation_steps=batches_validate,
                shuffle=False,  # Not needed, our generator shuffles everything already
                use_multiprocessing=False)  # This requires a thread-safe generator which we don't have
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
            pass

        print('Training completed')

    def save(self, filename):
        """ Save the weights of this model. """
        self.model.save_weights(filename)

    def load(self, filename):
        """ Load the weights for this model from a file. """
        self.model.load_weights(filename)

    def classify(self, X):
        """
        Classify a set of samples. This method should be called after successful training and loading of the model.
        :param X: Full-size image which to classify.
        :return: A list where every element denotes the normalized probability that a sample is road
        """
        # Subdivide the images into blocks
        img_patches = create_patches(X, self.PATCH_SIZE, 16, self.PADDING)

        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            img_patches = np.rollaxis(img_patches, 3, 1)

        # Run prediction
        Z = self.model.predict(img_patches)

        # Normalize probabilities
        Z = Z[:, 1] / (Z[:, 0] + Z[:, 1])

        Z = group_patches(Z, X.shape[0])

        Z = np.reshape(Z, (int(X.shape[1]/self.PATCH_SIZE), int(X.shape[2]/self.PATCH_SIZE)))

        return Z
