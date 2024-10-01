import cv2 as cv
import numpy as np
import moderngl as mgl
from logger import LoggerGenerator
import matplotlib.pyplot as plt
import sys
from constants import GLOBAL_CONSTANTS as constants
from graph import Graph, lineIntersection, segmentIntersection, makeGraphFromLines
from util import *

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

def _toRange(v, min, max, newmin, newmax):
            if max == min:
                return (v-min)*(newmax-newmin) + newmin
            return (v - min)*(newmax - newmin)/(max-min)+newmin

def _polarToLine(rho, theta):
    """
    Converts vector to line to two points, which are off screen."""
    max_rho, min_rho, max_theta, min_theta = np.sqrt(600*600+400*400), -np.sqrt(600*600+400*400), np.pi, 0
    if rho < min_rho or rho > max_rho or theta < min_theta or theta > max_theta:
        return None, None
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1800*(-b)), int(y0 + 1800*(a)))
    pt2 = (int(x0 - 1800*(-b)), int(y0 - 1800*(a)))
    return pt1, pt2
    

def drawHoughLines(overlay, lines, linewidth = 1):
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        pt1, pt2 = _polarToLine(rho, theta)
        if pt1 is None or pt2 is None:
            continue
        cv.line(overlay, pt1, pt2, (0,0,255,50), linewidth, cv.LINE_AA)

def drawHoughBuckets(overlay, lines):
    max_rho, min_rho, max_theta, min_theta = np.sqrt(600*600+400*400), -np.sqrt(600*600+400*400), np.pi, 0
    if lines is None:
        return
    
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        pt1, pt2 = _polarToLine(rho, theta)
        if pt1 is None or pt2 is None:
            continue
        cv.circle(overlay,(int(_toRange(theta,min_theta,max_theta,0,600)), int(_toRange(rho,min_rho,max_rho,0,400))), 2, (255,255,0,255))


fps = 0.0
def drawOverlays(app, overlay):
    ctx = app.ctx
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

    overlay = np.zeros((canny.shape[0],canny.shape[1],4),dtype=np.uint8)
    if app.SHOW_HOUGH:
        lines = cv.HoughLines(canny, 1, np.pi / 180, constants.opencv.HOUGH_THRESH, None, 0, 0)
        drawHoughEdges(overlay, canny)
        drawHoughBuckets(overlay, lines)
    else:
        overlay = canny

    drawOverlays(app, overlay)

def exportFbo(data_fbo, output_file = "output.png"):
    image = _fboToImage(data_fbo)
    image_8bit = (image * 255).astype(np.uint8)
    cv.imwrite(output_file, image_8bit)

def postProcessImage(file):
    image = cv.imread(file)
    canny = genCannyFromFrameBuffer(image)
    overlay = np.zeros((canny.shape[0],canny.shape[1],4),dtype=np.uint8)
    #drawHoughEdges(overlay, canny)
    lines = cv.HoughLines(canny, 1, np.pi / 180, 60, None, 0, 0)
    drawHoughLines(overlay, lines)
    cv.imshow("canny", cv.addWeighted(image, 1, overlay[:,:,0:3], 0.2, 0))

def lsd(file):
    image = cv.imread(file)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lsd = cv.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]
    drawn = lsd.drawSegments(image, lines)
    cv.imshow("lsd", drawn)

def prob(file):
    # Get Probabilistic Hough Lines from the image
    image = cv.imread(file)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 5, 150, apertureSize=3)
    cv.imshow("canny",edges)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=10)
    lines = lineMatrixToPairs(lines)

    lines = combineParallelLines(lines)
    graph = makeGraphFromLines(lines)
    graph.draw_graph(image)
    
    cv.imshow("prob", image)

def lsd_intersections(file):
    # Get LSD Lines from the image
    image = cv.imread(file)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lsd = cv.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]
    lines = lineMatrixToPairs(lines)

    lines = combineParallelLines(lines)
    # Find all intersections between line pairs and draw them
    for i in range(0, len(lines)):
        for j in range(i+1, len(lines)):
            a, b = lines[i]
            c, d = lines[j]
            pt = segmentIntersection(a, b, c, d, threshold=10)
            if pt is not None:
                cv.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), 2)
            
    # Draw all the lines in random colors
    for i in range(0, len(lines)):
        a, b = lines[i]
        a = np.array(a, dtype=int)
        b = np.array(b, dtype=int)
        cv.line(image, tuple(a), tuple(b), (int(np.random.randint(0, 256)), int(np.random.randint(0, 256)), int(np.random.randint(0, 256))), 1)
    
    cv.imshow("lsd intersections", image)

def prob_intersections(file):
    # Get Probabilistic Hough Lines from the image
    image = cv.imread(file)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 5, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=10)
    lines = lineMatrixToPairs(lines)

    lines = combineParallelLines(lines)
    # Find all intersections between line pairs
    for i in range(0, len(lines)):
        for j in range(i+1, len(lines)):
            a, b = lines[i]
            c, d = lines[j]
            pt = segmentIntersection(a, b, c, d, threshold=10)
            if pt is not None:
                cv.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), 2)
    
    # Draw all the lines in random colors
    for i in range(0, len(lines)):
        a, b = lines[i]
        a = np.array(a, dtype=int)
        b = np.array(b, dtype=int)
        cv.line(image, tuple(a), tuple(b), (int(np.random.randint(0, 256)), int(np.random.randint(0, 256)), int(np.random.randint(0, 256))), 1)
    
    cv.imshow("prob intersections", image)

if __name__ == "__main__":
    file = "sc_cube.png"
    lsd_intersections(file)
    prob_intersections(file)
    prob(file)
    lsd(file)

    cv.waitKey(0)
    cv.destroyAllWindows()