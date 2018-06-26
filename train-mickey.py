#!/usr/bin/env python

from mickey import *

EPOCH_COUNT = 40

# model = Inceptuous("Inceptuous-opt-test")
# model = InceptionResNet()
model = ResNet(full_preactivation=False)
#model = RedNet(full_preactivation=False)
# model = MyModel("NasNetLarge")
# model.load('weights-Inception-ResNet-v2-e014-f1-0.6475.hdf5')
model.model.summary()
model.train(EPOCH_COUNT)
model.save('model_weights.h5')
