import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import rawpy
from pyunraw import PyUnraw


parser = argparse.ArgumentParser()
parser.add_argument('input', help='48 Bit TiFF with corresponding IR file, OR Silverfast 64 bit DNG file')
parser.add_argument('-c', '--max-coverage', type=float, default=0.003)
parser.add_argument('-b', '--border-buffer', type=float, default=5.0)
parser.add_argument('-a', '--algorithm', type=str, default='telea')
parser.add_argument('-g', '--gamma', type=float, default=2.2, help="Adjust gamma of input")
parser.add_argument('--write-mask', action='store_true')
parser.add_argument('--write-gamma', action='store_true')


args = parser.parse_args()

input_path: str = args.input

BORDER_BUFFER = args.border_buffer
INPAINT_RADIUS = 0.95
MAX_COVERAGE = args.max_coverage
COMPRESSION_PARAMS = [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_DEFLATE]

# Write TIFF w IR channel
if input_path.endswith(".dng"):
    ir_path = input_path + '.ir.tiff'
    unraw = PyUnraw(input_path)
    unraw.write_sixteen_bits(True, True)
    unraw.set_output_format(True)
    unraw.unraw(1, ir_path)
    raw = rawpy.imread(input_path)
    rgb_img = cv2.cvtColor(raw.raw_image, cv2.COLOR_RGB2BGR)
    ir_img = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
elif input_path.endswith(".tif") or input_path.endswith(".tiff"):
    ir_path = input_path.replace(".tif", "-ir.tif")
    rgb_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    # TODO detect IR file type here
    # For now we assume it's 16 bit color
    ir_img = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("ir-tmp.tif", ir_img)
else:
    print(f"Unrecognized file extension: {input_path}")
    sys.exit(-1)


output_basename = os.path.splitext(os.path.basename(input_path))[0]
parent_dir = Path(input_path).parent.absolute()
processed_dir = parent_dir / "processed"
cleaned_dir = processed_dir / "cleaned"
mask_dir = processed_dir / "mask"
gamma_dir = processed_dir / "gamma"
cleaned_path = cleaned_dir / (output_basename + "-cleaned.tiff")
mask_path = mask_dir / (output_basename + "-mask.tiff")
gamma_path = gamma_dir / (output_basename + "-gamma.tiff")


y_buf = int(ir_img.shape[0]*(BORDER_BUFFER / 100))
x_buf = int(ir_img.shape[1]*(BORDER_BUFFER / 100))


ir_img_white = ir_img.max()

prev_mask_coverage = None
thresh_mask = None
cutoff = 0
threshold = 0
for threshold in range(50, 90):
    # percent of most white value
    cutoff = ir_img_white * (threshold * 0.01)
    # B&W mask as 8-bit mono (covert to bool, then to uint8, then mult by 255)
    temp_mask = (ir_img < cutoff).astype(np.uint8) * (2**8-1)
    # only consider buffered mask for coverage testing
    buffered_mask = temp_mask[y_buf:-y_buf, x_buf:-x_buf]
    mask_coverage = np.count_nonzero(buffered_mask) / buffered_mask.size
    # print(f"threshold: {threshold} cutoff: {cutoff} mask coverage: {mask_coverage}")
    if mask_coverage > MAX_COVERAGE:
        break
    prev_mask_coverage = mask_coverage
    thresh_mask = temp_mask
    # mask_path = f"{ir_path}.mask.{threshold}.tiff"
    # cv2.imwrite(mask_path, thresh_mask)

print(f"threshold: {threshold-1} coverage: {prev_mask_coverage}")

# sys.exit(-1)

if args.write_mask:
    mask_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(mask_path, thresh_mask, COMPRESSION_PARAMS)


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = ((np.arange(0, 65536) / 65535) ** inv_gamma) * 65535
    table = table.astype(np.uint16)
    return table[image]

if args.gamma != 1.0:
    for channel in range(3):
        rgb_img[:,:,channel] = adjust_gamma(rgb_img[:,:,channel], args.gamma)
    if args.write_gamma:
        gamma_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(gamma_path, rgb_img, COMPRESSION_PARAMS)


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

cleaned_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(cleaned_path, cleaned_rgb_img, params=COMPRESSION_PARAMS)

# TODO delete ir temp file if we used unraw
