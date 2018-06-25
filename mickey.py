from cnn_model import *
from keras.layers import Activation, Concatenate

class Inceptuous(CnnModel):
    def __init__(self, model_name):
        super().__init__()
        self.MODEL_NAME = model_name

    def build_model(self):
        layers = BasicLayers(relu_version='parametric')

        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor
        x = layers.cbr(x, 32, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        # Half-size, 32 features
        x1 = layers.cbr(x, 32, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x2 = layers._max_pool(x, pool=(3, 3), strides=(2, 2), padding='same')
        x = Concatenate(axis=1)([x1, x2])
        x = layers._spatialdropout(x)
        # Half-size, 64 features
        x1 = layers.cbr(x, 64, kernel_size=(1, 1))
        x1 = layers.cbr(x1, 64, kernel_size=(3, 3), strides=(2, 2))
        x2 = layers.cbr(x, 64, kernel_size=(1, 1))
        x2 = layers.cbr(x2, 64, kernel_size=(1, 5))
        x2 = layers.cbr(x2, 64, kernel_size=(5, 1))
        x2 = layers.cbr(x2, 64, kernel_size=(3, 3), strides=(2, 2))
        x = Concatenate(axis=1)([x1, x2])
        x = layers._spatialdropout(x)
        # Half-size, 128 features
        x1 = layers.cbr(x, 128, kernel_size=(3, 3), strides=(2, 2))
        x2 = layers._max_pool(x, pool=(3, 3))
        x = Concatenate(axis=1)([x1, x2])
        x = layers._spatialdropout(x)
        # Half-size, 256 features
        x1 = layers.cbr(x, 256, kernel_size=(1, 1))
        x1 = layers.cbr(x1, 256, kernel_size=(3, 3), strides=(2, 2))
        x2 = layers.cbr(x, 256, kernel_size=(1, 1))
        x2 = layers.cbr(x2, 256, kernel_size=(1, 3))
        x2 = layers.cbr(x2, 256, kernel_size=(3, 1))
        x2 = layers.cbr(x2, 256, kernel_size=(3, 3), strides=(2, 2))
        x = Concatenate(axis=1)([x1, x2])
        x = layers._spatialdropout(x)
        # Half-size, 512 features

        x = layers._flatten(x)
        x = layers._dense(x, 2 * ((self.CONTEXT * self.CONTEXT) // (self.PATCH_SIZE * self.PATCH_SIZE)))
        x = layers._act_fun(x)
        x = layers._dense(x, self.NB_CLASSES)  # Returns a logit
        x = Activation('softmax')(x)  # No logit anymore
        self.model = Model(inputs=input_tensor, outputs=x)


class InceptionResNet(CnnModel):
    def __init__(self):
        super().__init__(context=256, batch_size=32, model_name="Inception-ResNet-v2")

    def build_model(self):
        incres = InceptionResNetLayer(relu_version='parametric', half_size=False)
        # incres = InceptionResNetLayer()
        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor

        x = incres.stem(x)

        for i in range(5):
            x = incres.block16(x)

        x = incres.block7(x)
        x = incres._act_fun(x)

        for i in range(10):
            x = incres.block17(x)

        x = incres.block18(x)
        x = incres._act_fun(x)

        for i in range(5):
            x = incres.block19(x)

        x = incres.cbr(x, 1024, (1, 1))
        x = incres.cbr(x, 256, (1, 1))

        x = incres._flatten(x)
        x = incres._dense(x, 2 * ((self.CONTEXT * self.CONTEXT) // (self.PATCH_SIZE * self.PATCH_SIZE)))
        x = incres._act_fun(x)
        x = incres._dense(x, self.NB_CLASSES)  # Returns a logit
        x = Activation('softmax')(x)  # No logit anymore
        self.model = Model(inputs=input_tensor, outputs=x)


class ResNet(CnnModel):
    FULL_PREACTIVATION = False

    def __init__(self, model_name, full_preactivation=False):
        super().__init__()
        self.MODEL_NAME = model_name
        self.FULL_PREACTIVATION = full_preactivation

    def build_model(self):
        resnet = ResNetLayers(relu_version='parametric', full_preactivation=self.FULL_PREACTIVATION)

        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor
        x = resnet.stem(x)
        for i, layers in enumerate(resnet.REPETITIONS_SMALL):
            for j in range(layers):
                x = resnet.vanilla(x, resnet.FEATURES[i], (j == 0))

        x = resnet._flatten(x)
        x = resnet._dense(x, 2 * ((self.CONTEXT * self.CONTEXT) // (self.PATCH_SIZE * self.PATCH_SIZE)))
        x = resnet._act_fun(x)
        x = resnet._dense(x, self.NB_CLASSES)  # Returns a logit
        x = Activation('softmax')(x)  # No logit anymore
        self.model = Model(inputs=input_tensor, outputs=x)
