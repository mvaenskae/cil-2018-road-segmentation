#!/usr/bin/env python

from mickey import *
from helpers import *
import datetime

d = datetime.datetime.utcnow()  # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"
predict_on = "test_images"

model = ResNet(full_preactivation=False)
model.load('model-weights_resnet18.h5')
model.model.summary()

submission_filename = 'submission-' + timestamp + '_' + model.MODEL_NAME + '.csv'

post_processing = False
generate_submission(model, os.path.join("data", predict_on), submission_filename, post_processing)
generate_overlay_images(model, os.path.join("data", predict_on), post_processing)
