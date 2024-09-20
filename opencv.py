import cv2 as cv
import numpy as np
import moderngl as mgl
from logger import LoggerGenerator
import matplotlib.pyplot as plt
import sys
from constants import GLOBAL_CONSTANTS as constants

def genCannyFromFrameBuffer(image):

    # cv.imshow("text", image)
    
    ### CALCULATIONS ###
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def get_edges_image(image, blur = (5, 5),thresh_soft = 5, thresh_hard = 10):
        blurred = cv.GaussianBlur(image, blur, 0)
        return cv.Canny(cv.normalize(blurred, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8), thresh_soft, thresh_hard)

    return get_edges_image(gray, thresh_soft=constants.opencv.CANNY_THRESH_SOFT, thresh_hard=constants.opencv.CANNY_THRESH_HARD)

def drawHoughEdges(overlay, canny):
    if(constants.opencv.HOUGH_PROB_LINE_WIDTH <= 0):
        return
    #image = cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)])
    lines = cv.HoughLinesP(canny, 1, np.pi/180, threshold=constants.opencv.HOUGH_PROB_THRESH,
                            minLineLength=constants.opencv.HOUGH_PROB_LINE_MIN_LENGTH,
                            maxLineGap=constants.opencv.HOUGH_PROB_LINE_MAX_GAP)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(overlay, (x1, y1), (x2, y2), (255, 0, 0,255), constants.opencv.HOUGH_PROB_LINE_WIDTH)

def drawHoughBuckets(overlay, canny):
    if(constants.opencv.HOUGH_LINE_WIDTH <= 0 and not constants.opencv.HOUGH_SHOW_COORDINATES):
        return
    lines = cv.HoughLines(canny, 1, np.pi / 180, constants.opencv.HOUGH_THRESH, None, 0, 0)
    
    np.set_printoptions(threshold=sys.maxsize)
    cv.circle(overlay, (600,400), 5, (255,255,255,255))
    if lines is not None:
        def toRange(v, min, max, newmin, newmax):
            if max == min:
                return (v-min)*(newmax-newmin) + newmin
            return (v - min)*(newmax - newmin)/(max-min)+newmin
        max_rho, min_rho, max_theta, min_theta = np.sqrt(600*600+400*400), -np.sqrt(600*600+400*400), np.pi, 0
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            if rho < min_rho or rho > max_rho or theta < min_theta or theta > max_theta:
                print("out of bounds: ", rho, theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1800*(-b)), int(y0 + 1800*(a)))
            pt2 = (int(x0 - 1800*(-b)), int(y0 - 1800*(a)))
            if constants.opencv.HOUGH_LINE_WIDTH > 0:
                cv.line(overlay, pt1, pt2, (0,0,255,50), constants.opencv.HOUGH_LINE_WIDTH, cv.LINE_AA)
            if constants.opencv.HOUGH_SHOW_COORDINATES:
                cv.circle(overlay,(int(toRange(theta,min_theta,max_theta,0,600)), int(toRange(rho,min_rho,max_rho,0,400))), 2, (255,255,0,255))


fps = 0.0
def drawCanny(app, canny):
    ctx = app.ctx
   
    lines = None
    overlay = np.zeros((canny.shape[0],canny.shape[1],4),dtype=np.uint8)

    if app.SHOW_HOUGH:
       drawHoughEdges(overlay, canny)
       drawHoughBuckets(overlay, canny)

    global fps
    fps = app.clock.get_fps()
    cv.putText(overlay,"fps: " + str(round(fps,2)),(0,50),cv.FONT_HERSHEY_PLAIN,1,(0,0,0,255), 2)

    ### DRAW ON SCREEN ###
    buffer = overlay.tobytes()
    app.buffers.opencv_tex.write(buffer)
    ctx.enable_only(ctx.BLEND)
    app.buffers.opencv_tex.use()
    app.mesh.vaos['blit'].render()

def _fboToImage(fbo : mgl.Framebuffer):
    """
    Reads the image data from a modernGL framebuffer
    """
    buffer = fbo.read(components=3,dtype="f4")
    raw = np.frombuffer(buffer,dtype="f4")
    image = raw.reshape((fbo.height,fbo.width,3))[::-1,:,::-1] # Shape properly and reverse the order
    return image

def postProcessFbo(app, data_fbo = None):
    if data_fbo is None:
        data_fbo = app.ctx.screen
    image = _fboToImage(data_fbo)
    canny = genCannyFromFrameBuffer(image)
    drawCanny(app, canny)