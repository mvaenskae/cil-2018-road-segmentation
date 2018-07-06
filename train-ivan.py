#!/usr/bin/env python

from ivan import *
import datetime

d = datetime.datetime.utcnow()  # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"

EPOCH_COUNT = 300
model_filename = 'model-weights_' + timestamp + '.h5'

model = SegNet("SegNet")
model.model.summary()
model.train(EPOCH_COUNT,'weights-SegNet-e074-ce-0.3171.hdf5',101)
model.save(model_filename)
