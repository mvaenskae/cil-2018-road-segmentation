#!/usr/bin/env python

from mickey import *
from helpers import *
import datetime

d = datetime.datetime.utcnow()  # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"
predict_on = "test_images"

model = RedNet(full_preactivation=False)
model.load('model-weights_rednet50.h5')
model.model.summary()

submission_directory = 'prediction-' + timestamp + '_' + model.MODEL_NAME

post_processing = False
generate_submission_heatmaps(model, os.path.join("data", predict_on), submission_directory, post_processing)
