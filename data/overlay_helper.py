from ..helpers import load_image, img_float_to_uint8, get_files_in_dir
from PIL import Image
import numpy as np

TEST_DIR = 'test_images/'
PREDICTION_DIR = 'predictions_rednet_e3600/'


filenames_test = [TEST_DIR + s for s in sorted(get_files_in_dir(TEST_DIR))]
filenames_preds = list(filter(lambda s: '.png' in s, [PREDICTION_DIR + s for s in sorted(get_files_in_dir(PREDICTION_DIR))]))
default_length = len(filenames_test)

overlay_list = []
for i in range(default_length):
    img_test = load_image(filenames_test.pop())
    img_pred = load_image(filenames_preds.pop())
    color_mask = np.zeros((img_test.shape[0], img_test.shape[1], 3), dtype=np.uint8)
    color_mask[:, :, 0] = img_pred * 255
    img_test8 = img_float_to_uint8(img_test)
    background_img = Image.fromarray(img_test8, 'RGB').convert("RGBA")
    overlay_img = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    blended = Image.blend(background_img, overlay_img, 0.3)
    overlay_list.append(blended)