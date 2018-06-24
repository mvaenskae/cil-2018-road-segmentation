#!/usr/bin/env python

from jimmy import *
from helpers import *
import datetime

d = datetime.datetime.utcnow() # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"
predict_on = "test_images"

model = VEGGG("VEGGG")
model.load('model_weights.h5')
model.model.summary()

submission_filename = 'submission-' + timestamp + '_' + predict_on + '.csv'

generate_submission(model, 'data/test_images', submission_filename, True)
generate_overlay_images(model, 'data/test_images', True)
