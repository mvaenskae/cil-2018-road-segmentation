from full_cnn import FullCNN
from keras import Model
from keras.layers import Input, UpSampling2D, Lambda
from keras_helpers import BasicLayers
from keras.activations import softmax


class SegNet(FullCNN):
    FULL_PREACTIVATION = False

    def __init__(self):
        super().__init__(image_size=608, batch_size=4, model_name="SegNet")

    def build_model(self):
        layers = BasicLayers(relu_version=self.RELU_VERSION)
        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor
        # SegNet
        # ENCODER
        # 13 Conv layes of VGG
        # PART 1
        x = layers.cbr(x, 64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')

        # PART 2
        x = layers.cbr(x, 128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')

        # PART 3
        x = layers.cbr(x, 256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')

        # PART 4
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')

        # PART 5
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')

        # DECODER
        # PART 1
        x = UpSampling2D(size=(2, 2), data_format="channels_first")(x)
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')

        # PART 2
        x = UpSampling2D(size=(2, 2), data_format="channels_first")(x)
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')

        # PART 3
        x = UpSampling2D(size=(2, 2), data_format="channels_first")(x)
        x = layers.cbr(x, 256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')

        # PART 4
        x = UpSampling2D(size=(2, 2), data_format="channels_first")(x)
        x = layers.cbr(x, 128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers.cbr(x, 64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')

        # PART 5
        x = UpSampling2D(size=(2, 2), data_format="channels_first")(x)
        x = layers.cbr(x, 64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers._conv2d(x, 2, 1, strides=(1, 1), dilation_rate=(1, 1), padding='same')
        x = layers._batch_norm(x)
        x = Lambda(lambda x: softmax(x, axis=1))(x)
        self.model = Model(inputs=input_tensor, outputs=x)