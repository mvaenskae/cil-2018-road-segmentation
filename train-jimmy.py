#!/usr/bin/env python

from jimmy import *

EPOCH_COUNT = 40

model = Simple("Simple")
model.model.summary()
model.train(EPOCH_COUNT)
model.save('simple_weights.h5')
