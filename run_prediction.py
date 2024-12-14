import argparse
from opencv import *
from opencv_renewed import *
parser = argparse.ArgumentParser(
                    prog='Cube detection project',
                    description='This file runs the many pipelines of the project, see help for the different options',
                    epilog='If something fails, you may try running the individual files (main.py, opencv.py, opencv_fit_color.py, opencv_points.py)')

parser.add_argument('image_name')           # positional argument
parser.add_argument('-d', '--detector')      # option that takes a value
parser.add_argument("-p", "--pipeline")
parser.add_argument('-v', '--verbose',
                    action='store_true')  # on/off flag

try:
    args = parser.parse_args()
    image_name = args.image_name
    detector = args.detector
    verbose = args.verbose
    pipeline = args.pipeline
except:
    # No file name passed
    print("<Warning> No file provided, using a default 'generated_cubes/sc_pres.png")
    print("<Warning> Additional parameters will also not be processed.")

    image_name = 'generated_images/demo_scarce.png'
    detector = None
    verbose = False
    pipeline = None

image = cv.imread(image_name)
if image is None:
    print("[Error] Could not open or find the image ", image_name)
    exit()

if detector is None:
    detector = 'lsd'
if pipeline is None:
    pipeline = 'graph'

if detector.lower() == 'lsd':
    lines = lsd(image)
elif detector.lower() == 'hough' or detector.lower() == 'houghp'.lower() or detector == 'prob':
    lines = prob(image, verbose)

if pipeline.lower() == 'graph':
    drawGraphPipeline(image.copy(), lines, verbose, verbose, True, False ,True)
elif pipeline.lower() == 'vanishing_points' or pipeline.lower() == 'lines' or pipeline.lower() == 'vp':
    drawFocalPointsPipeline(image, lines)
elif pipeline.lower() == 'mixed':
    drawMixedPipeline(image,lines)
else:
    print("[Error] Invalid pipeline name: ", pipeline)
    print("<info> The options are 'graph' for the graphing pipeline, 'vanishing' / 'lines' / 'vp' for the vanishing points pipeline")
    exit()