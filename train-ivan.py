#!/usr/bin/env python
from ivan import *

EPOCH_COUNT = 40

model = SegNet("SegNet")
model.model.summary()
model.train(EPOCH_COUNT)
model.save('model_weights.h5')
