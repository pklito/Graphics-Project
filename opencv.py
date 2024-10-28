import cv2 as cv
import numpy as np
import moderngl as mgl
from logger import LoggerGenerator
import matplotlib.pyplot as plt
import sys
from constants import GLOBAL_CONSTANTS as constants
from graph import *
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

def postProcessCubesFbo(app, data_fbo = None):
    if data_fbo is None:
        data_fbo = app.ctx.screen
    image = _fboToImage(data_fbo)
    image = (image * 255).astype(np.uint8)
    trans = lsd(image, 2, scale=0.5, dont_display=True)
    


def exportFbo(data_fbo, output_file = "output.png"):
    image = _fboToImage(data_fbo)
    image_8bit = (image * 255).astype(np.uint8)
    cv.imwrite(output_file, image_8bit)

def _getIntersections(lines):
    """
    lines are in the format rho, theta
    """
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]
            if (abs(rho1- rho2) < 40 and abs(theta1-theta2) < np.deg2rad(10)) or  abs(theta1-theta2) > np.deg2rad(80):
                continue
            A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
            b = np.array([rho1, rho2])
            try:
                x0, y0 = np.linalg.solve(A, b)
                intersections.append((x0,y0))
            except:
                pass # Lines are parallel
    return intersections

def _processIntersections(image, intersections):
    

    overlay = np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
    for intersection in intersections:
        if intersection[0] < 0 or intersection[0] > image.shape[1] or intersection[1] < 0 or intersection[1] > image.shape[0]:
            continue
        cv.circle(overlay, (int(intersection[0]), int(intersection[1])), 5, (0,55,255))

    return cv.addWeighted(image, 1, overlay[:,:,0:3], 0.8, 0)

def postProcessImage(image):
    canny = genCannyFromFrameBuffer(image)
    overlay = np.zeros((canny.shape[0],canny.shape[1],4),dtype=np.uint8)
    #drawHoughEdges(overlay, canny)
    lines = cv.HoughLines(canny, 1, np.pi / 180, 50, None, 0, 0)
    drawHoughLines(overlay, lines)
    image = cv.addWeighted(image, 1, overlay[:,:,0:3], 0.2, 0)

    intersections = _getIntersections(lines)
    image = _processIntersections(image, intersections)

    cv.imshow("canny", image)

def lsd(image, detector = 0, scale = 0.8, sigma_scale = 0.6, quant = 2.0, ang_th = 22.5, log_eps = 0.0, density_th = 0.7, n_bins = 1024, post_process = True, dont_display = False):
    print(image.dtype)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lsd = cv.createLineSegmentDetector(detector, scale=scale, sigma_scale=sigma_scale, quant=quant, ang_th=ang_th, log_eps=log_eps, density_th=density_th, n_bins=n_bins)
    lines = lsd.detect(gray)[0]
    if not dont_display:
        drawn = lsd.drawSegments(image, lines)
    lines = lineMatrixToPairs(lines)
    # for line in lines:
    #     cv.line(image, line[0], line[1], (0,255,0), 2)
    
    lines = combineParallelLines(lines)
    graph = makeGraphFromLines(lines)
    if not post_process:
        graph.draw_graph(image, (0,0,255), (0,255,0), 2, 5)
        cv.imshow("lsd plain", image)
        return
    graph = mergeOverlappingVertices(graph, threshold=9, neighbor_limit=1)

    #graph.draw_graph(image, (0,0,255), (0,255,0), 2, 5)
    graph = connectIntersectingEdges(graph, threshold_splice=0, threshold_detect=7)
    graph = mergeOverlappingVertices(graph, threshold=5, merge_neighbors=True)
    graph = mergeOverlappingVertices(graph, threshold=8, merge_neighbors=True)

    faces = getFaces(graph)
    
    trans = handleFaces(image, faces, not dont_display)
    if not dont_display:
        graph.draw_graph(image, vertex_numbers=True)
        cv.imshow("lsd full" + str(np.random.random()), drawn)
    
    return trans

def pointToScreen(rvec,tvec, world_point, camera_matrix = None):
    tvec = np.array(tvec)
    rvec = np.array(rvec)
    world_point = np.array(world_point)
    if camera_matrix is None:
        camera_matrix = np.array([
            [300, 0, 300],
            [0, 300, 200],
            [0, 0, 1]
        ])
    rotation_matrix, _ = cv.Rodrigues(rvec)
    rel_coord = np.dot(rotation_matrix, world_point.T) + tvec
    imaginary_point = (camera_matrix @ (rel_coord)).flatten()
    return imaginary_point[:2]/imaginary_point[2]

def handleFaces(image, faces, draw = True):
    object_points = np.array([[-0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,-0.5,0.5]], dtype=np.float32)
    object_points_inv = np.array([[-0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,0.5,-0.5],[0.5,-0.5,-0.5]], dtype=np.float32)


    # Define the camera matrix for a perspective camera with resolution 600x400 and FOV of 90 degrees
    focal_length_x = 600 / 2
    focal_length_y = 600 / 2
    camera_matrix = np.array([
        [focal_length_x, 0, 300],
        [0, focal_length_y, 200],
        [0, 0, 1]
    ])

    
    trans = []
    index = 0
    for face in faces:
        index += 1
        if np.cross(face[1] - face[0], face[2] - face[0]) < 0:
            face = face[::-1]
        image_points = np.array(face, dtype=np.float32)
        ret, rvec, tvec = cv.solveP3P(object_points, image_points, camera_matrix, None, flags=cv.SOLVEPNP_P3P)
        rvec, tvec = rvec[0], tvec[0]
        if not ret:
            continue

        rotation_matrix, _ = cv.Rodrigues(rvec)
        
        world_point1 = np.array([[0, 0, 0.5]], dtype=np.float32)
        world_point2 = np.array([[0, 0, -0.5]], dtype=np.float32)
        rel_coord1 = np.dot(rotation_matrix, world_point1.T) + tvec
        rel_coord2 = np.dot(rotation_matrix, world_point2.T) + tvec
        rel_coord = tvec
        imaginary_point = (camera_matrix @ (rel_coord)).flatten()

        print([round(i,2) for i in (rel_coord).flatten()], [round(i,2) for i in (tvec).flatten()])
            # Define the point in the world coordinates
        
        cv.circle(image, tuple(np.array(imaginary_point[:2]/imaginary_point[2], dtype=int)), 3, (255,255,255), -1)
        # cv.drawFrameAxes(image, camera_matrix, None, rvec, tvec, 0.5)
        
        trans.append([[round(a,2) for a in rvec.ravel()], [round(a,2) for a in tvec.ravel()]])
    
    # shape = np.array([[-0.5,-0.5,0],[-0.5,0.5,0],[0.5,0.5,0],[0.5,-0.5,0]], dtype=np.float32)
    if draw:
        for rvec, tvec in trans[:]:
            # cv.polylines(image, [np.array([pointToScreen(rvec, tvec, point, camera_matrix) for point in shape], dtype=np.int32)], True, (0,255,0), 2)
            tvec = np.array(tvec)
            rvec = np.array(rvec)
            cv.drawFrameAxes(image, camera_matrix, None, rvec, tvec, 0.5)
    return trans
            



def prob(image):
    # Get Probabilistic Hough Lines from the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 5, 150, apertureSize=3)
    cv.imshow("canny",edges)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=10)
    lines = lineMatrixToPairs(lines)
    # for line in lines:
    #     cv.line(image, line[0], line[1], (0,255,0), 2)
    
    lines = combineParallelLines(lines)
    graph = makeGraphFromLines(lines)
    graph = mergeOverlappingVertices(graph, threshold=10, neighbor_limit=1)
    #graph.draw_graph(image, (0,0,255), (0,255,0), 2, 5)
    graph = connectIntersectingEdges(graph, threshold_splice=0, threshold_detect=7)
    graph = mergeOverlappingVertices(graph, threshold=5, merge_neighbors=True)
    graph = mergeOverlappingVertices(graph, threshold=7, merge_neighbors=True)

    faces = getFaces(graph)

    handleFaces(image, faces)
    graph.draw_graph(image, vertex_numbers=True)
    
    cv.imshow("prob", image)

if __name__ == "__main__":
    file = "sc_rgb.png"
    image = cv.imread(file)
    lsd(image.copy(),2,scale=0.5)
    lsd(image.copy(),2,scale=0.5,post_process=False)
    postProcessImage(image.copy())



    cv.waitKey(0)
    cv.destroyAllWindows()