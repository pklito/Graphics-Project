import argparse
from opencv import *
parser = argparse.ArgumentParser(
                    prog='Cube detection project',
                    description='This file runs the many pipelines of the project, see help for the different options',
                    epilog='If something fails, you may try running the individual files (main.py, opencv.py, opencv_fit_color.py, opencv_points.py)')

parser.add_argument('image_name')           # positional argument
parser.add_argument('-d', '--detector')      # option that takes a value
parser.add_argument('-v', '--verbose',
                    action='store_true')  # on/off flag

print("trying")
try:
    args = parser.parse_args()
    image_name = args.image_name
    detector = args.detector
    verbose = args.verbose
except:
    print("<Warning> No file provided, using a default 'generated_cubes/sc_pres.png")
    image_name = 'generated_images/demo_scarce.png'
    detector = 'lsd'
    verbose = False

image = cv.imread(image_name)

if image is None:
    print("[Error] Could not open or find the image ", image_name)
    exit()

drawGraphPipeline(image.copy(), lsd(image), False, False, True, False ,True)

cv.waitKey(0)
cv.destroyAllWindows()