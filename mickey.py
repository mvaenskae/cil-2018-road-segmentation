from label_cnn import LabelCNN
from full_cnn import FullCNN
import keras.applications.nasnet
from keras import Model
from keras.layers import Activation, Concatenate, Add, Input, Cropping2D
from keras_helpers import BasicLayers, ResNetLayers, InceptionResNetLayer, RedNetLayers

class Inceptuous(LabelCNN):
    def __init__(self, model_name='Inceptuous'):
        super().__init__(model_name=model_name)

    def build_model(self):
        layers = BasicLayers(relu_version=self.RELU_VERSION)

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
        x = layers._dense(x, 2 * ((self.IMAGE_SIZE * self.IMAGE_SIZE) // (self.PATCH_SIZE * self.PATCH_SIZE)))
        x = layers._act_fun(x)
        x = layers._dense(x, self.NB_CLASSES)  # Returns a logit
        x = Activation('softmax')(x)  # No logit anymore
        self.model = Model(inputs=input_tensor, outputs=x)


class InceptionResNet(LabelCNN):
    def __init__(self):
        super().__init__(image_size=128, batch_size=32, model_name="Inception-ResNet-v2")

    def build_model(self):
        incres = InceptionResNetLayer(relu_version=self.RELU_VERSION, half_size=False)
        # incres = InceptionResNetLayer()
        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor

        x = incres.stem(x)

        for i in range(2):
            x = incres.block16(x)

        x = incres.block7(x)
        x = incres._act_fun(x)

        for i in range(5):
            x = incres.block17(x)

        #x = incres.block18(x)
        #x = incres._act_fun(x)

        #for i in range(2):
        #    x = incres.block19(x)

        #x = incres.cbr(x, 1024, (1, 1))
        #x = incres.cbr(x, 256, (1, 1))

        x = incres._flatten(x)
        x = incres._dense(x, 6 * ((self.IMAGE_SIZE * self.IMAGE_SIZE) // (self.PATCH_SIZE * self.PATCH_SIZE)))
        x = incres._dropout(x, 0.5)
        x = incres._act_fun(x)
        x = incres._dense(x, self.NB_CLASSES)  # Returns a logit
        x = Activation('softmax')(x)  # No logit anymore
        self.model = Model(inputs=input_tensor, outputs=x)


class ResNet(LabelCNN):
    FULL_PREACTIVATION = False

    def __init__(self, full_preactivation=False):
        super().__init__(image_size=72, batch_size=64, relu_version='parametric', model_name="ResNet")
        self.FULL_PREACTIVATION = full_preactivation

    def build_model(self):
        rednet = RedNetLayers(relu_version=self.RELU_VERSION, full_preactivation=self.FULL_PREACTIVATION)

        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor
        x, _ = rednet.stem(x)
        for i, layers in enumerate(rednet.REPETITIONS_SMALL):
            for j in range(layers):
                x = rednet.short(x, rednet.FEATURES[i], (j == 0))

        x = rednet._flatten(x)
        x = rednet._dense(x, 6 * ((self.IMAGE_SIZE * self.IMAGE_SIZE) // (self.PATCH_SIZE * self.PATCH_SIZE)))
        x = rednet._dropout(x, 0.5)
        x = rednet._act_fun(x)
        x = rednet._dense(x, self.NB_CLASSES)  # Returns a logit
        x = Activation('softmax')(x)  # No logit anymore
        self.model = Model(inputs=input_tensor, outputs=x)


class RedNet(FullCNN):
    FULL_PREACTIVATION = False

    def __init__(self, model_name="RedNet", full_preactivation=False):
        super().__init__(image_size=608, batch_size=2, model_name=model_name)
        self.FULL_PREACTIVATION = full_preactivation

    def build_model(self):
        rednet = RedNetLayers(relu_version=self.RELU_VERSION, full_preactivation=self.FULL_PREACTIVATION)
        agent_list = []
        agent_layers = []

        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor
        x, a0 = rednet.stem(x)
        #agent_layers.append(rednet.agent_layer(a0, rednet.FEATURES[0]))
        for i, layers in enumerate(rednet.REPETITIONS_NORMAL):
            if i == 0:
                for j in range(layers):
                    x = rednet.vanilla(x, rednet.FEATURES[i], (j == 0))
            else:
                for j in range(layers):
                    x = rednet.vanilla_down(x, rednet.FEATURES[i], (j == 0))
            #agent_list.append(x)

        #agent_layers = []
        #for i in range(len(agent_list)):
        #    agent_layers.append(rednet.agent_layer(agent_list[i], rednet.FEATURES[i]))

        #x = agent_layers.pop()

        for i, layers in enumerate(rednet.REPETITIONS_UP_NORMAL):
            for j in range(layers):
                x = rednet.residual_up(x, rednet.FEATURES_UP[i], (j == layers - 1))
            if i + 1 != len(rednet.REPETITIONS_UP_NORMAL):
                # Remove this for full-size images. Needed for 304x304
                if i == 0 and self.IMAGE_SIZE <= 304:
                    x = Cropping2D(cropping=((0, 1), (0, 1)), data_format=self.DATA_FORMAT)(x)
                #x = Add()([x, agent_layers.pop()])

        x = rednet.last_block(x)

        x = rednet._tcbr(x, self.NB_CLASSES, kernel_size=(2, 2), strides=(2, 2))
        x = Activation('softmax')(x)  # No logit anymore
        self.model = Model(inputs=input_tensor, outputs=x)


class SimpleNet(LabelCNN):

    def __init__(self, model_name='SimpleNet'):
        super().__init__(image_size=72, batch_size=64, relu_version='parametric', model_name=model_name)

    def build_model(self):
        layers = BasicLayers(relu_version=self.RELU_VERSION)
        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor

        x = layers.cbr(x, 64, kernel_size=(3, 3))

        for i in range(3):
            x = layers.cbr(x, 128, kernel_size=(3, 3))
        x = layers._max_pool(x, pool=(2, 2))

        for i in range(2):
            x = layers.cbr(x, 128, kernel_size=(3, 3))

        x = layers.cbr(x, 128, kernel_size=(3, 3))
        x = layers._max_pool(x, pool=(2, 2))

        for i in range(2):
            x = layers.cbr(x, 128, kernel_size=(3, 3))
        x = layers._max_pool(x, pool=(2, 2))

        x = layers.cbr(x, 128, kernel_size=(1, 1))

        x = layers.cbr(x, 128, kernel_size=(1, 1))
        x = layers._max_pool(x, pool=(2, 2))

        x = layers.cbr(x, 128, kernel_size=(3, 3))
        x = layers._max_pool(x, pool=(2, 2))

        x = layers._flatten(x)
        x = layers._dense(x, 6 * ((self.IMAGE_SIZE * self.IMAGE_SIZE) // (self.PATCH_SIZE * self.PATCH_SIZE)))
        x = layers._dropout(x, 0.5)
        x = layers._act_fun(x)
        x = layers._dense(x, self.NB_CLASSES)  # Returns a logit
        x = Activation('softmax')(x)  # No logit anymore
        self.model = Model(inputs=input_tensor, outputs=x)

class EasyNet(LabelCNN):
    def __init__(self, model_name='EasyNet'):
        super().__init__(image_size=72, batch_size=64, relu_version='parametric', model_name=model_name)


    def build_model(self):
        layers = BasicLayers(relu_version=self.RELU_VERSION)
        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor

        x = layers.cbr(x, 64, (5, 5))
        x = layers._max_pool(x, pool=(2, 2))
        x = layers._spatialdropout(x, 0.25)

        x = layers.cbr(x, 128, (3, 3))
        x = layers._max_pool(x, pool=(2, 2))
        x = layers._spatialdropout(x, 0.25)

        x = layers.cbr(x, 256, (3, 3))
        x = layers._max_pool(x, pool=(2, 2))
        x = layers._spatialdropout(x, 0.25)

        x = layers.cbr(x, 512, (3, 3))
        x = layers._max_pool(x, pool=(2, 2))
        x = layers._spatialdropout(x, 0.25)

        x = layers._flatten(x)
        x = layers._dense(x, 6 * ((self.IMAGE_SIZE * self.IMAGE_SIZE) // (self.PATCH_SIZE * self.PATCH_SIZE)))
        x = layers._dropout(x, 0.5)
        x = layers._act_fun(x)
        x = layers._dense(x, self.NB_CLASSES)  # Returns a logit
        x = Activation('softmax')(x)  # No logit anymore
        self.model = Model(inputs=input_tensor, outputs=x)
