import cv2 as cv
import numpy as np

i = 0
def opencv_process(image_rgb):
    global i
    if i == 0:
        i+=1
        return
    cv.imshow("name",image_rgb)
    cv.waitKey(0)
    image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blue_channel, green_channel, red_channel = cv.split(image)
    # Apply GaussianBlur to reduce noise and improve contour detection

    def get_edges_image(image, blur = (5, 5),thresh1 = 5, thresh2 = 10):
        blurred = cv.GaussianBlur(image, blur, 0)
        return cv.Canny(blurred, thresh1, thresh2)

    canny_split = cv.cvtColor(cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)]),cv.COLOR_BGR2GRAY)
    canny = get_edges_image(gray)
    #image = cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)])
    lines = cv.HoughLinesP(canny, 1, np.pi/180, threshold=60, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image