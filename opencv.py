import cv2 as cv
import numpy as np
import moderngl as mgl
from logger import LoggerGenerator
import matplotlib.pyplot as plt
import sys
from constants import GLOBAL_CONSTANTS as constants

def genCannyFromContext(ctx : mgl.Context):
     # Get the screen as a buffer
    buffer = ctx.screen.read(components=3,dtype="f4")
    raw = np.frombuffer(buffer,dtype="f4")
    image = raw.reshape((ctx.screen.height,ctx.screen.width,3))[::-1,:,::-1]
    
    ### CALCULATIONS ###
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def get_edges_image(image, blur = (5, 5),thresh_soft = 5, thresh_hard = 10):
        blurred = cv.GaussianBlur(image, blur, 0)
        return cv.Canny(cv.normalize(blurred, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8), thresh_soft, thresh_hard)

    return get_edges_image(gray, thresh_soft=10, thresh_hard=30)

def drawHoughEdges(overlay, canny):
    #image = cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)])
    lines = cv.HoughLinesP(canny, 1, np.pi/180, threshold=60, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(overlay, (x1, y1), (x2, y2), (255, 0, 0,255), constants.opencv.HOUGH_PROB_LINE_WIDTH)

def drawHoughBuckets(overlay, canny):
    lines = cv.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)
    
    np.set_printoptions(threshold=sys.maxsize)
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
            cv.line(overlay, pt1, pt2, (0,0,255,255), constants.opencv.HOUGH_LINE_WIDTH, cv.LINE_AA)
    

fps = 0.0
def opencv_process(app):
    ctx = app.ctx
    lines = None
    overlay = np.zeros((ctx.screen.height,ctx.screen.width,4),dtype=np.uint8)

    if app.SHOW_HOUGH:
       canny = genCannyFromContext(ctx)
       drawHoughEdges(overlay, canny)
       drawHoughBuckets(overlay, canny)

    global fps
    fps = app.clock.get_fps()
    cv.putText(overlay,"fps: " + str(round(fps,2)),(0,50),cv.FONT_HERSHEY_PLAIN,1,(0,0,0,255), 2)

    ### DRAW ON SCREEN ###
    buffer = overlay.tobytes()
    app.mesh.textures['opencv'].write(buffer)
    ctx.enable_only(ctx.BLEND)
    app.mesh.textures['opencv'].use()
    app.mesh.vaos['opencv'].render()
    
