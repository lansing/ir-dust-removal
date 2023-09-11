import argparse
import os

import cv2
import numpy as np
import rawpy
from pyinpaint import Inpaint
from pyunraw import PyUnraw


parser = argparse.ArgumentParser()
parser.add_argument('input', help='Silverfast 64 bit DNG file')
parser.add_argument('-c', '--max-coverage', type=float, default=0.002)
parser.add_argument('-b', '--border-buffer', type=float, default=5.0)
parser.add_argument('-g', '--gamma', type=float, default=2.2)
parser.add_argument('-m', '--write-mask', action='store_true')
parser.add_argument('-f', '--subfolder', type=str)
parser.add_argument('-a', '--algorithm', type=str, default='telea')


args = parser.parse_args()

input_path = args.input

BORDER_BUFFER = args.border_buffer
INPAINT_RADIUS = 0.95
MAX_COVERAGE = args.max_coverage
MAX_COVERAGE_JUMP = 1.8 # not currently used


# Write TIFF w IR channel
ir_path = input_path + '.ir.tiff'
unraw = PyUnraw(input_path)
unraw.write_sixteen_bits(True, True)
unraw.set_output_format(True)
unraw.unraw(1, ir_path)

raw = rawpy.imread(input_path)
rgb_img = cv2.cvtColor(raw.raw_image, cv2.COLOR_RGB2BGR)
ir_img = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)

y_buf = int(ir_img.shape[0]*(BORDER_BUFFER / 100))
x_buf = int(ir_img.shape[1]*(BORDER_BUFFER / 100))

prev_mask_coverage = None
thresh_mask = None
for threshold in range(50, 90):
    cutoff = ir_img.max() * threshold * 0.01
    temp_mask = (ir_img < (ir_img.max() * threshold*0.01)).astype(np.uint8) * (2**8-1)
    buffered_mask = temp_mask[y_buf:-y_buf, x_buf:-x_buf]
    mask_coverage = np.count_nonzero(buffered_mask) / buffered_mask.size
    if mask_coverage > MAX_COVERAGE:
        break
    prev_mask_coverage = mask_coverage
    thresh_mask = temp_mask
    # mask_path = f"{ir_path}.mask.{threshold}.tiff"
    # cv2.imwrite(mask_path, thresh_mask)

print(f"threshold: {threshold-1} coverage: {prev_mask_coverage}")

if args.write_mask:
    mask_path = f"{ir_path}.mask.{threshold}.tiff"
    cv2.imwrite(mask_path, thresh_mask)

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = ((np.arange(0, 65536) / 65535) ** inv_gamma) * 65535
    table = table.astype(np.uint16)
    return table[image]

cleaned_rgb_img = np.zeros(rgb_img.shape).astype(np.uint16)

if args.algorithm == 'telea':
    INPAINT_ALGO = cv2.INPAINT_TELEA
    for channel in range(3):
        channel_img = rgb_img[:,:,channel]
        cleaned_channel = cv2.inpaint(channel_img, thresh_mask, INPAINT_RADIUS, INPAINT_ALGO)
        cleaned_rgb_img[:,:,channel] = cleaned_channel
elif args.algorithm == 'biharmonic':
    from skimage.restoration import inpaint
    cleaned_rgb_img_float = inpaint.inpaint_biharmonic(rgb_img, thresh_mask, channel_axis=-1)
    cleaned_rgb_img = (cleaned_rgb_img_float * 65535).astype(np.uint16)

if args.gamma != 1.0:
    for channel in range(3):
        cleaned_rgb_img[:,:,channel] = adjust_gamma(cleaned_rgb_img[:,:,channel], args.gamma)

if args.subfolder:
    output_dir = os.path.join(os.path.dirname(input_path), args.subfolder)
    if not os.path.exist(output_dir):
        os.mkdir(output_dir)
else:
    output_dir = os.path.dirname(input_path)

output_basename = os.path.splitext(os.path.basename(input_path))[0]
output_path = os.path.join(output_dir, output_basename + "-cleaned.tiff")

cv2.imwrite(output_path, cleaned_rgb_img, params=(cv2.IMWRITE_TIFF_COMPRESSION, 32946))

os.remove(ir_path)
