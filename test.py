import cv2 as cv
import numpy as np

# Load the image
image = cv.imread('test3.png', cv.IMREAD_COLOR)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blue_channel, green_channel, red_channel = cv.split(image)
# Apply GaussianBlur to reduce noise and improve contour detection

def get_edges_image(image, blur = (5, 5),thresh1 = 5, thresh2 = 10):
    blurred = cv.GaussianBlur(image, blur, 0)
    return cv.Canny(blurred, thresh1, thresh2)

blue_new = get_edges_image(blue_channel)
green_new = get_edges_image(green_channel)
red_new = get_edges_image(red_channel)

lines = cv.HoughLines(cv.cvtColor(cv.merge([blue_new,red_new,green_new]), cv.COLOR_BGR2GRAY), 1, np.pi / 180, 150, None, 0, 0)
print(lines)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(image, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

# Display the result
cv.imshow('Contours', image)
cv.waitKey(0)
cv.destroyAllWindows()
