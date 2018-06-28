#!/usr/bin/env python

from mickey import *
import datetime

d = datetime.datetime.utcnow()  # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"

EPOCH_COUNT = 1000
model_filename = 'model-weights_' + timestamp + 'h5'

#model = InceptionResNet()
#model = ResNet(full_preactivation=False)
model = RedNet(full_preactivation=False)
#model = SimpleNet()
# model.load('weights-Inception-ResNet-v2-e014-f1-0.6475.hdf5')
model.model.summary()
model.train(EPOCH_COUNT)
model.save(model_filename)
