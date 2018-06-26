#!/usr/bin/env python

from mickey import *

EPOCH_COUNT = 40

model = ResNet("ResNet")
model.model.summary()
model.train(EPOCH_COUNT)
model.save('resnet_non_stop_smoothed_weights.h5')
