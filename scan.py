import argparse
import os
import sys

DEFAULT_BASE_DIR = "/Users/max/Pictures/Scans/Raw"

parser = argparse.ArgumentParser()
parser.add_argument('output', type=str, help="Output directory name")
parser.add_argument('frame', type=int, help='Frame number')
parser.add_argument('-b', '--base-dir', default=DEFAULT_BASE_DIR)
parser.add_argument('-g', '--gamma', type=float, default=2.2)
parser.add_argument('-d', '--dust-removal', action='store_true')
parser.add_argument('-a', '--dust-algorithm', type=str, default='telea')
parser.add_argument('-c', '--dust-max-coverage', type=float, default=0.002)


args = parser.parse_args()

frame = args.frame

cmd = 'scan'

while True:
    
    if cmd == 'exit':
        sys.exit(0)
        
    if cmd == 'next':
        frame += 1

    filename = args.output + "-" + str(frame).zfill(2) + ".tif"
    out_path = os.path.join(args.base_dir, args.output, filename)

    print(f"Scan -> {out_path}")

    cmd = None
    while not cmd:
        next_cmd = input("n (next), r (retry), x (exit): ")

        if next_cmd.startswith('n') or next_cmd == '':
            cmd = 'next'
        elif next_cmd.startswith('x' or 'e'):
            cmd = 'exit'
        elif next_cmd.startswith('r'):
            cmd = 'scan'
        else:
            print(f"Unknown command: {next_cmd} ... must be one of: n, r, x")

