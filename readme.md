# Traditional cube detecting graphics project

## Writeup
Read [here](present/graphics%20write%20up%20-%20Full.pdf) the full writeup

## Video
https://youtu.be/lJQKC-Njt0E

## Installation

The project uses python 3. To install the libraries, enter the main directory and run

`pip install -r requirements.txt`
or alternatively
`pip install pygame opencv-python numpy moderngl matplotlib`

# Running the code
There are two seperate runnable files, the first is the simulator that can generate cubes for you.
The second is the prediction that takes an image and guesses where the cubes are.

## Graphics simulator
```sh
py run_simulation.py
```

Controls:
ESC - exit
P - Pause
WASD - move around
Q/E - move down/up
Mouse - turn camera
T - export current frame as 'generated_images/output.png'
Y - Print camera matrix and cube positions
G - Process the current frame directly

The area has some randomly placed cubes I used to test out different configurations for the image.
Pressing G does the LSD graph method of finding cubes without having to run `run_prediction.py`.
*Note*: To continue, you may have to close the result MatPlotLib provides.

## Predictor
### Predictor
`py run_prediction.py <image_name> [-d DETECTOR] [-v]`

This script processes an image to detect cubes using different detection methods.

#### Arguments:
- `image_name`: The name of the image file to process.
- `-d`, `--detector`: The edge detection method to use. Options are `lsd` (default), `hough`.
- `-p`, `--pipeline` Which analysis will be run on the image.
  - For the first faces and solvePNP solution, `graph`
  - For the method which localizes edges individually using the vanishing points, `lines`
  - For the mixture, which gets the edges from the vanishing points and uses solvePNP with faces of those edges, use `mixed`
- `-v`, `--verbose`: If set, enables verbose mode for detailed images along with the result.

#### Example usages:
```sh
py run_prediction.py generated_images/demo_scarce.png
```
```sh
py run_prediction.py generated_images/demo_scarce.png --detector lsd --verbose --pipeline graph
```

```sh
py run_prediction.py generated_images/demo_scarce.png --pipeline lines
```
If no image is provided, the script defaults to `generated_images/demo_scarce.png` with the `lsd` detector and non-verbose mode.

#### Detection Methods:
- `lsd`: Line Segment Detector.
- `hough`: Standard Hough Transform.
- `houghp`: Probabilistic Hough Transform.
- `prob`: Alias for Probabilistic Hough Transform.
