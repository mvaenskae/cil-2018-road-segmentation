#!/usr/bin/env python

from mickey import *
import datetime

d = datetime.datetime.utcnow()  # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"

EPOCH_COUNT = 25
model_filename = 'model-weights_resnet18.h5'

model = ResNet(full_preactivation=False)
model.model.summary()
model.train(EPOCH_COUNT)
model.save(model_filename)
