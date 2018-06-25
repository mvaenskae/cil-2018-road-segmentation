#!/usr/bin/env python

from mickey import *
from helpers import *
import datetime

d = datetime.datetime.utcnow() # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"
predict_on = "test_images"

model = InceptionResNet("Inception-ResNet-v2")
model.load('model_weights.h5')
model.model.summary()


submission_filename = 'submission-' + timestamp + '_' + predict_on + '.csv'

post_processing = False
generate_submission(model, 'data/test_images', submission_filename, post_processing)
