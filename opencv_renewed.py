import cv2 as cv
import numpy as np
import moderngl as mgl
from logger import LoggerGenerator
import matplotlib.pyplot as plt
import sys
from graph import *
from util import *
from opencv_points import matsToCubes, plot_cubes, alignTrans
from opencv_fit_color import *
from opencv import lsd, prob, drawGraphPipeline, drawEdges
import itertools
#
# New pipeline
#
def edges_to_polar_lines(edges):
    return np.array([(np.sign(np.arctan2(b[0] - a[0], b[1] - a[1]))*(a[1]*b[0]-b[1]*a[0])/np.linalg.norm(b-a), np.fmod(-np.arctan2(b[0] - a[0], b[1] - a[1]) + np.pi,np.pi)) for a, b in edges])

def classifyEdges(edges, threshold_multiplier = 1.2):
    lines = edges_to_polar_lines(edges)

    # Get camera angles
    phi_theta, loss = regress_lines(lines, iterations=1000, refinement_iterations=500, refinement_area=np.deg2rad(15))
    phi, theta = phi_theta
    focal_points = get_focal_points(phi, theta)

    pairs = [(edge, which_line(focal_points, line, threshold = loss * threshold_multiplier)) for edge, line in zip(edges, lines)]
    x_edges = [line for line, which in pairs if which == "x"]
    y_edges = [line for line, which in pairs if which == "y"]
    z_edges = [line for line, which in pairs if which == "z"]
    
    return x_edges, y_edges, z_edges, phi, theta

def smoothEdges(x_edges,y_edges,z_edges):
    x_edges = combineParallelLines(x_edges)
    y_edges = combineParallelLines(y_edges)
    z_edges = combineParallelLines(z_edges)
    return x_edges, y_edges, z_edges

def get_faces_from_pairs(edges1, edges2, threshold = 15):
    faces = []
    indices = []
    for i in range(len(edges1)):
        e1 = edges1[i]
        for j in range(len(edges2)):
            e2 = edges2[j]
            if segments_distance(*e1, *e2) > threshold:
                continue
            for k in range(i + 1, len(edges1)):
                e3 = edges1[k]
                if segments_distance(*e2, *e3) > threshold:
                    continue
                for l in range(j + 1, len(edges2)):
                    e4 = edges2[l]
                    if segments_distance(*e3, *e4) > threshold or segments_distance(*e4, *e1) > threshold:
                        continue

                    # Eliminate duplicate faces:
                    indices.append((i, j, k, l))
                    faces.append([lineIntersection(*e1, *e2), lineIntersection(*e2, *e3), lineIntersection(*e3, *e4), lineIntersection(*e4, *e1)])         
    new_faces = []
    banned_faces = []
    for a, edge_numbers, face in zip(range(len(faces)), indices, faces):
        if edge_numbers in banned_faces:
            continue
        accepted = True
        for b, edge_numbers2, face2 in list(zip(range(len(faces)), indices, faces))[a+1:]:
            # Two faces share two edges.
            if (edge_numbers[0], edge_numbers[2]) == (edge_numbers2[0], edge_numbers2[2]) or (edge_numbers[1], edge_numbers[3]) == (edge_numbers2[1], edge_numbers2[3]):
                # They overlap (significantly)
                if pointInConvexPolygon(sum([np.array(f) for f in face])/4, face2) or pointInConvexPolygon(sum([np.array(f) for f in face2])/4, face):
                    
                    if faceCircumference(face) > faceCircumference(face2):
                        accepted = False
                        break
                    else:
                        banned_faces.append(face2)
        if accepted:
            new_faces.append(face)

    # Orient clockwise
    new_new_faces = []
    for face in new_faces:
        e1 = [face[1][0] - face[0][0], face[1][1] - face[0][1]]
        e2 = [face[2][0] - face[1][0], face[2][1] - face[1][1]]
        if e1[0]*e2[0] - e1[1]*e2[0] < 0:
            new_new_faces.append(face)
        else:
            new_new_faces.append(face[::-1])
    return new_new_faces

def drawFaces(image, faces, color, shrink_factor = 0.75):
    for face in faces:
        center = sum(np.array(p) for p in face)/4
        cv.line(image, np.array(face[0],dtype=np.uint32), np.array(face[1],dtype=np.uint32), (0.5*color[0],0.5*color[1],0.5*color[2]), 3)
        cv.line(image, np.array(face[1],dtype=np.uint32), np.array((np.array(face[2])+face[1])/2,dtype=np.uint32), (0.5*color[0],0.5*color[1],0.5*color[2]), 3)
        cv.fillPoly(image, [np.asarray([shrink_factor * np.array(p) + (1-shrink_factor) * center for p in face],dtype=np.int32)], color)

def drawMats(image, mats):
    for rvec, tvec in mats:
        cv.drawFrameAxes(image, getIntrinsicsMatrix(), None, rvec, tvec, 1)
    cv.imshow("Mats"+str(np.random.rand()), image)

def drawFocalPointsPipeline(image, edges):
    original_image = image.copy()
    # # # Colored edges drawing # # #
    x_edges, y_edges, z_edges, phi, theta = classifyEdges(edges, 1.2)
    #image = cv.addWeighted(image, 0.5, np.zeros(image.shape, image.dtype), 0.5, 0)
    drawEdges(image, x_edges, (0, 0, 200),1)
    drawEdges(image, y_edges, (0, 100, 0),1)
    drawEdges(image, z_edges, (100, 0, 0),1)

    cv.imshow("Focal points", image)

    # # # MatPlotLib sine wave drawing # # #
    draw_vanishing_points_plots(edges_to_polar_lines(edges), phi, theta, show=False)

    image = original_image.copy()
    # # # Connected graph drawing # # #
    x_edges, y_edges, z_edges = smoothEdges(x_edges, y_edges, z_edges)
    drawEdges(image, x_edges, (0, 0, 255),3)
    drawEdges(image, y_edges, (0, 255, 0),3)
    drawEdges(image, z_edges, (255, 0, 0),3)

    zfaces=get_faces_from_pairs(x_edges, y_edges)
    yfaces=get_faces_from_pairs(z_edges, x_edges)
    xfaces=get_faces_from_pairs(y_edges, z_edges)
    
    drawFaces(image, xfaces, (0, 0, 255))
    drawFaces(image, yfaces, (0, 255, 0))
    drawFaces(image, zfaces, (255, 0, 0))
    cv.imshow("Connected Edges", image)
    
    drawMats(original_image.copy(), handleClassifiedFaces(phi, theta, zfaces, "z", 9000000))
    drawMats(original_image.copy(), handleClassifiedFaces(phi, theta, xfaces, "x", 9000000))
    drawMats(original_image.copy(), handleClassifiedFaces(phi, theta, yfaces, "y", 9000000))
    drawEdgeNumbers(original_image.copy(), x_edges, y_edges, z_edges)
    plt.show()

def drawEdgeNumbers(image, x_edges, y_edges, z_edges ):
    for i, edge in enumerate(x_edges):
        cv.putText(image, str(i), (int((edge[0][0] + edge[1][0])/2), int((edge[0][1] + edge[1][1])/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
    for i, edge in enumerate(y_edges):
        cv.putText(image, str(i), (int((edge[0][0] + edge[1][0])/2), int((edge[0][1] + edge[1][1])/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
    for i, edge in enumerate(z_edges):
        cv.putText(image, str(i), (int((edge[0][0] + edge[1][0])/2), int((edge[0][1] + edge[1][1])/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 0), 1)
    cv.imshow("Edge numbers", image)

def handleClassifiedFaces(phi, theta, zfaces, axis, LENGTH = 9000000):
    def randomsign():
        return 1 if np.random.rand() > 0.5 else -1
    if axis == "z":
        shape = [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]
    elif axis == "y":
        shape = [[0,0,0],[1,0,0],[1,0,1],[0,0,1]]
    else:
        shape = [[0,0,0],[0,0,1],[0,1,1],[0,1,0]]
    camera_matrix = getIntrinsicsMatrix()
    focal_points = get_focal_points(phi, theta)
    length = LENGTH
    object_points = np.array([[-length,0,0],[0,length,0],[0,0,length]] + shape, dtype=np.float32)
    mats = []
    for face in zfaces:
        for iter in range(50):
            # Face needs to be clockwise!!
            image_points = [focal_points + face]
            rand_object_point =  np.array([[randomsign()*a for a in p] for p in [[-length,0,0],[0,length,0],[0,0,length]]] + shape,dtype=np.float32)
            rand_image_point = image_points[::randomsign()]
            offset = np.random.randint(0,3)
            rand_image_point = np.array(image_points[offset:] + image_points[:offset],dtype=np.float32)
            ret, rvec, tvec = cv.solvePnP(rand_object_point, rand_image_point, camera_matrix, None)
            tvec = np.array(tvec)
            rvec = np.array(rvec)
            if np.linalg.norm(tvec) < 1000 and np.linalg.norm(tvec) > 0.01:
                break
        mats.append((rvec, tvec))

    return mats

def justMatPlotPipeline(image, edges):
    phi_theta, loss = regress_lines(edges_to_polar_lines(edges), iterations=1000, refinement_iterations=500, refinement_area=np.deg2rad(15))
    phi, theta = phi_theta
    draw_vanishing_points_plots(edges_to_polar_lines(edges), phi, theta, show=False)
    plt.show()

def facesToTrans(xfaces, yfaces,zfaces, phi, theta):
    xmats = handleClassifiedFaces(phi, theta, xfaces, "x")
    ymats = handleClassifiedFaces(phi, theta, yfaces, "y")
    zmats = handleClassifiedFaces(phi, theta, zfaces, "z")
    def transformPoint(point, rvec, tvec):
        return cv.Rodrigues(rvec)[0] @ np.array(point) + tvec
    points = []
    for mat in xmats:
        points.append(max(transformPoint([0.5,0.5,0.5], *mat), transformPoint([-0.5,0.5,0.5], *mat), key = lambda x: np.linalg.norm(x)))
        
    for mat in ymats:
        points.append(max(transformPoint([0.5,0.5,0.5], *mat), transformPoint([0.5,-0.5,0.5], *mat), key = lambda x: np.linalg.norm(x)))
    for mat in zmats:
        points.append(max(transformPoint([0.5,0.5,0.5], *mat), transformPoint([0.5,0.5,-0.5], *mat), key = lambda x: np.linalg.norm(x)))

    return points


def getCubesVP(edges):
    x_edges, y_edges, z_edges, phi, theta = classifyEdges(edges, 1.2)
    x_edges, y_edges, z_edges = smoothEdges(x_edges, y_edges, z_edges)
    zfaces=get_faces_from_pairs(x_edges, y_edges)
    yfaces=get_faces_from_pairs(x_edges, z_edges)
    xfaces=get_faces_from_pairs(z_edges, y_edges)
    centers = facesToTrans(xfaces, yfaces, zfaces, phi, theta)
    return centers

if __name__ == "__main__":
    file = "sc_bugged.png"
    image = cv.imread(file)
    drawFocalPointsPipeline(image, lsd(image))

    cv.waitKey(0)
    cv.destroyAllWindows()