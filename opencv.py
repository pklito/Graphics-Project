import cv2 as cv
import numpy as np
import moderngl as mgl
from logger import LoggerGenerator
import matplotlib.pyplot as plt
import sys
from constants import GLOBAL_CONSTANTS as constants
from graph import *
from util import *
from opencv_points import matsToCubes, plot_cubes, alignTrans

#
#   Old post processing, hough lines, with constants.json
#

WIDTH = 600
HEIGHT = 400

def doCanny(image):
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


def drawHoughLines(overlay, lines, linewidth = 1):
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        pt1, pt2 = polarToLine(rho, theta)
        if pt1 is None or pt2 is None:
            continue
        cv.line(overlay, pt1, pt2, (0,0,255,50), linewidth, cv.LINE_AA)


def drawHoughBuckets(overlay, lines):
    max_rho, min_rho, max_theta, min_theta = np.sqrt(WIDTH*WIDTH+HEIGHT*HEIGHT), -np.sqrt(WIDTH*WIDTH+HEIGHT*HEIGHT), np.pi, 0
    if lines is None:
        return
    
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        pt1, pt2 = polarToLine(rho, theta)
        if pt1 is None or pt2 is None:
            continue
        cv.circle(overlay,(int(toRange(theta,min_theta,max_theta,0,WIDTH)), int(toRange(rho,min_rho,max_rho,0,HEIGHT))), 2, (255,255,0,255))


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


def postProcessImage(image):
    canny = doCanny(image)
    overlay = np.zeros((canny.shape[0],canny.shape[1],4),dtype=np.uint8)
    #drawHoughEdges(overlay, canny)
    lines = cv.HoughLines(canny, 1, np.pi / 180, 50, None, 0, 0)
    drawHoughLines(overlay, lines)
    image = cv.addWeighted(image, 1, overlay[:,:,0:3], 0.2, 0)

    intersections = _getIntersections(lines)
    image = _overlayIntersections(image, intersections)

    cv.imshow("canny", image)

#
#   FBO functions
#

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
    canny = doCanny(image)

    overlay = np.zeros((canny.shape[0],canny.shape[1],4),dtype=np.uint8)
    if app.SHOW_HOUGH:
        lines = cv.HoughLines(canny, 1, np.pi / 180, constants.opencv.HOUGH_THRESH, None, 0, 0)
        drawHoughEdges(overlay, canny)
        drawHoughBuckets(overlay, lines)
    else:
        overlay = canny

    drawOverlays(app, overlay)

def cubesToWorld(cubes, camera):
    rounded_yaw = - np.round(2*camera.yaw/np.pi,0)
    rounded_yaw = rounded_yaw * np.pi/2
    cubes = [[c[0]*np.cos(rounded_yaw), c[1], np.sin(rounded_yaw)*c[2]] for c in cubes]

    return [[np.floor(x1)+np.floor(x2) for x1,x2 in zip(c,camera.position)] for c in cubes]


def postProcessCubesFbo(app, data_fbo = None, camera_trans = None, display = False, pipelineFunc = None):
    if pipelineFunc is None:
        pipelineFunc = getCubes
    if data_fbo is None:
        data_fbo = app.ctx.screen
    image = _fboToImage(data_fbo)
    image = (image * 255).astype(np.uint8)
    #image = cv.blur(image, (3,3))
    trans = pipelineFunc(lsd(image, 2, scale=0.5))
    if display:
        drawGraphPipeline(image.copy(), lsd(image, 2, scale=0.5), doGraph=False, doAxis=True, doFaces=True)
        drawGraphPipeline(image.copy(), lsd(image, 2, scale=0.5), doGraph=True, doAxis=False, doFaces=True)
        print("trans:", trans)
    if camera_trans is None:
        mats, excluded = alignTrans(trans, threshold=0.97)
        cubes = matsToCubes(mats)
    else:
        print("This shouldn't work (faces aren't oriented right)")
        cubes = [(np.linalg.inv(camera_trans) @ np.array([-x for x in t[1]] + [1])).ravel() for t in trans]
    
    if display:
        plot_cubes(cubes)
    return cubes


def exportFbo(data_fbo, output_file = "output.png"):
    image = _fboToImage(data_fbo)
    image_8bit = (image * 255).astype(np.uint8)
    cv.imwrite(output_file, image_8bit)


#
#   Graph intersections
#

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

def _overlayIntersections(image, intersections):
    overlay = np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
    for intersection in intersections:
        if intersection[0] < 0 or intersection[0] > image.shape[1] or intersection[1] < 0 or intersection[1] > image.shape[0]:
            continue
        cv.circle(overlay, (int(intersection[0]), int(intersection[1])), 5, (0,55,255))

    return cv.addWeighted(image, 1, overlay[:,:,0:3], 0.8, 0)

#
#   Edge detectors
#

def lsd(image, detector = 0, scale = 0.8, sigma_scale = 0.6, quant = 2.0, ang_th = 22.5, log_eps = 0.0, density_th = 0.7, n_bins = 1024):
    print(image.dtype)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lsd = cv.createLineSegmentDetector(detector, scale=scale, sigma_scale=sigma_scale, quant=quant, ang_th=ang_th, log_eps=log_eps, density_th=density_th, n_bins=n_bins)
    lines = lsd.detect(gray)[0]
    lines = lineMatrixToPairs(lines)
    return lines

def prob(image):
    # Get Probabilistic Hough Lines from the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 5, 150, apertureSize=3)
    cv.imshow("canny",edges)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=10)
    lines = lineMatrixToPairs(lines)
    return lines
    
#
#   Graph detector pipeline
#
def linesToPlanarGraph(lines):
    lines = combineParallelLines(lines)
    graph = makeGraphFromLines(lines)
    graph = mergeOverlappingVertices(graph, threshold=9, neighbor_limit=1)

    #graph.draw_graph(image, (0,0,255), (0,255,0), 2, 5)
    graph = connectIntersectingEdges(graph, threshold_splice=0, threshold_detect=7)
    graph = mergeOverlappingVertices(graph, threshold=5, merge_neighbors=True)
    graph = mergeOverlappingVertices(graph, threshold=8, merge_neighbors=True)
    return graph

def _pointToScreen(rotation_matrix,tvec, world_point, camera_matrix):
    rel_coord = np.dot(rotation_matrix, world_point.T) + tvec
    imaginary_point = (camera_matrix @ (rel_coord)).flatten()
    return imaginary_point[:2]/imaginary_point[2]

def pointToScreen(rvec,tvec, world_point, camera_matrix = None):
    tvec = np.array(tvec)
    rvec = np.array(rvec)
    world_point = np.array(world_point)
    if camera_matrix is None:
        camera_matrix = getIntrinsicsMatrix()
    rotation_matrix, _ = cv.Rodrigues(rvec)
    return _pointToScreen(rotation_matrix, tvec, world_point, camera_matrix)

def handleFaces(faces):
    """
    get 4 corners of faces and convert them to rvec, tvec pairs
    """
    object_points = np.array([[-0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,-0.5,0.5]], dtype=np.float32)
    object_points_inv = np.array([[-0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,0.5,-0.5],[0.5,-0.5,-0.5]], dtype=np.float32)


    # Define the camera matrix for a perspective camera with resolution 600x400 and FOV of 90 degrees

    camera_matrix = getIntrinsicsMatrix()

    
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
        
        trans.append([[round(a,2) for a in rvec.ravel()], [round(a,2) for a in tvec.ravel()]])
    return trans

#
#   Pipelines
#
def getCubes(lines):
    graph = linesToPlanarGraph(lines)
    faces = getFaces(graph)
    trans = handleFaces(faces)
    return trans

def drawFrameAxesMat(image, mat, tvec, camera_intrinsics, length = 1, width = 3):
    x = np.array([length, 0, 0])
    y = np.array([0, length, 0])
    z = np.array([0, 0, length])
    porigin = np.array(_pointToScreen(mat, tvec, np.array([0,0,0]), camera_intrinsics), dtype=np.int32)
    px = np.array(_pointToScreen(mat, tvec, x, camera_intrinsics), dtype=np.int32)
    py = np.array(_pointToScreen(mat, tvec, y, camera_intrinsics), dtype=np.int32)
    pz = np.array(_pointToScreen(mat, tvec, z, camera_intrinsics), dtype=np.int32)
    cv.line(image, porigin, px, (0,0,255), width)
    cv.line(image, porigin, py, (0,255,0), width)
    cv.line(image, porigin, pz, (255, 0, 0), width)



def drawGraphPipeline(image, lines, doGraph = True, doAxis = False, doFaces = False, doNewAxis = False, doCubes = False):
    graph = linesToPlanarGraph(lines)
    faces = getFaces(graph)
    trans = handleFaces(faces)
    mats, excluded_mats = alignTrans(trans)
    # cubes = matsToCubes(mats)

    print(trans)

    camera_matrix = getIntrinsicsMatrix()

    if doGraph:
        graph.draw_graph(image, edge_width=1,vertex_size=3)
    
    if doFaces:
        for face in faces:
            print(np.asarray(face,dtype=np.int32))
            cv.fillPoly(image, [np.asarray(face,dtype=np.int32)], (0,0,100))
            cv.polylines(image, [np.asarray(face,dtype=np.int32)], True, (0,0,255), 3)
    
    if doAxis:
        for rvec, tvec in trans:
            tvec = np.array(tvec)
            rvec = np.array(rvec)
            cv.drawFrameAxes(image, camera_matrix, None, rvec, tvec, 0.5)

    
    if doNewAxis:
        drawFixedAxes(trans, mats, excluded_mats)
        for mat, tvec in mats:
            drawFrameAxesMat(image, mat, tvec, camera_matrix, 0.5, 3)
        for mat, tvec in excluded_mats:
            pass
            # drawFrameAxesMat(image, mat, tvec, camera_matrix, 0.5, 3)

    if doCubes:
        points = matsToCubes(mats)
        plot_cubes(points)
    cv.imshow("drawPipeline" + str(np.random.randint(0,99)), image)

def drawEdges(image, edges, color = (255,255,255), width = 1):
    for edge in edges:
        cv.line(image, np.array(edge[0],dtype=np.uint32), np.array(edge[1],dtype=np.uint32), color, width)

def drawLines(image, lines, color = (255,255,255), dim_screen = 0.5):
    if dim_screen > 0:
        image = cv.addWeighted(image, 1-dim_screen, np.zeros(image.shape, image.dtype), dim_screen, 0)
    else:
        image = image.copy()
    drawEdges(image, lines, color)
    cv.imshow("lines", image)

def drawLinesColorful(image, lines, name = "lines"):
    image = cv.addWeighted(image, 0.5, np.zeros(image.shape, image.dtype), 0.5, 0)
    for line in lines:
        red = 255*np.random.random()
        blue = 255 - red
        green = 255*np.random.random()
        cv.line(image, np.array(line[0],dtype=np.uint32), np.array(line[1],dtype=np.uint32), (blue, green, red), 2)
    cv.imshow(name, image)

def draw_tris(ax, mat, r = 'r', g = 'g', b = 'b', offset = (0,0,0), linestyle = ''):
    o = offset
    ax.quiver(o[0], o[1], o[2], mat[0,0] + o[0],mat[1,0] + o[1],mat[2,0] + o[2],color=r, linestyle=linestyle)
    ax.quiver(o[0], o[1], o[2], mat[0,1] + o[0],mat[1,1] + o[1],mat[2,1] + o[2],color=g, linestyle=linestyle)
    ax.quiver(o[0], o[1], o[2], mat[0,2] + o[0],mat[1,2] + o[1],mat[2,2] + o[2],color=b, linestyle=linestyle)

import matplotlib.pyplot as plt
def drawFixedAxes(trans, mats, excluded_mats):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    
    for rvec, tvec in trans:
        mat = cv.Rodrigues(np.array(rvec))[0]
        draw_tris(ax2, mat)
        
    for mat, trans in mats:
        draw_tris(ax, mat)
    for mat, trans in excluded_mats:
        draw_tris(ax, mat, r = (0.5,0,0), g = (0,0.5,0), b = (0,0,0.5), linestyle='--')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reoriented rotation matrices')
    ax2.set_title('Input rotation matrices')


    plt.show()

if __name__ == "__main__":
    file = 'generated_images/demo_scarce.png'
    image = cv.imread(file)
    cv.imshow("base", image)
    drawGraphPipeline(image.copy(), lsd(image), True, False, False, True,True)
    drawLinesColorful(image,lsd(image), name = "lsd")
    # drawLinesColorful(image,prob(image), name = "prob")
    # postProcessImage(image.copy())

    cv.waitKey(0)
    cv.destroyAllWindows()