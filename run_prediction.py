import argparse

parser = argparse.ArgumentParser(
                    prog='Cube detection project',
                    description='This file runs the many pipelines of the project, see help for the different options',
                    epilog='If something fails, you may try running the individual files (main.py, opencv.py, opencv_fit_color.py, opencv_points.py)')

parser.add_argument('filename')           # positional argument
parser.add_argument('-c', '--count')      # option that takes a value
parser.add_argument('-v', '--verbose',
                    action='store_true')  # on/off flag

print("trying")
args = parser.parse_args()
print(args.filename, args.count, args.verbose)
