# -*- coding: utf-8 -*-

from keras import backend as K

class AbstractCNN:
    """
    Abstract base class for any CNN model.
    """
    def __init__(self,
                 image_size=112,
                 nb_classes=2,
                 batch_size=16,
                 model_name='AbstractBase',
                 data_format='channels_first',
                 relu_version=None,
                 leaky_relu_alpha=0.01):
        """ Construct a CNN classifier. """
        self.IMAGE_SIZE = image_size
        self.NB_CLASSES = nb_classes
        self.BATCH_SIZE = batch_size
        self.DATA_FORMAT = data_format
        self.RELU_VERSION = relu_version
        self.LEAKY_RELU_ALPHA = leaky_relu_alpha
        self.MODEL_NAME = model_name
        self.model = None
        self.images, self.groundtruth = None, None
        self.images_train, self.groundtruth_train = None, None
        self.images_validate, self.groundtruth_validate = None, None

        # Compatibility with Theano and Tensorflow ordering
        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            self.INPUT_SHAPE = (3, self.IMAGE_SIZE, self.IMAGE_SIZE)
        else:
            self.INPUT_SHAPE = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3)

        self.build_model()
    
    def build_model(self):
        """
        This method is required to be overwritten by a child-class and supposed to generate the network.
        :return: Nothing
        """
        raise NotImplementedError('AbstractCNN::build_model is not yet implemented.')

    def train(self, epochs, checkpoint=None):
        """
        Train this model.
        """
        raise NotImplementedError('AbstractCNN::train is not yet implemented.')

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
        """
        raise NotImplementedError('AbstractCNN::classify is not yet implemented.')
