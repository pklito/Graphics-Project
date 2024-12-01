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
    
    return x_edges, y_edges, z_edges

def smoothEdges(x_edges,y_edges,z_edges):
    print(len(x_edges), len(y_edges), len(z_edges))
    x_edges = combineParallelLines(x_edges)
    y_edges = combineParallelLines(y_edges)
    z_edges = combineParallelLines(z_edges)
    print(len(x_edges), len(y_edges), len(z_edges))
    return x_edges, y_edges, z_edges

def drawFocalPointsPipeline(image, edges):
    original_image = image.copy()
    # # # Colored edges drawing # # #
    x_edges, y_edges, z_edges = classifyEdges(edges, 1.2)
    #image = cv.addWeighted(image, 0.5, np.zeros(image.shape, image.dtype), 0.5, 0)
    drawEdges(image, x_edges, (0, 0, 200),2)
    drawEdges(image, y_edges, (0, 100, 0),2)
    drawEdges(image, z_edges, (100, 0, 0),2)

    cv.imshow("Focal points", image)

    # # # MatPlotLib sine wave drawing # # #
    # phi_theta, loss = regress_lines(edges_to_polar_lines(edges), iterations=1000, refinement_iterations=500, refinement_area=np.deg2rad(15))
    # phi, theta = phi_theta
    # draw_vanishing_points_plots(edges_to_polar_lines(edges), phi, theta)

    image = original_image.copy()
    # # # Connected graph drawing # # #
    x_edges, y_edges, z_edges = smoothEdges(x_edges, y_edges, z_edges)
    drawEdges(image, x_edges, (0, 0, 200),2)
    drawEdges(image, y_edges, (0, 100, 0),2)
    drawEdges(image, z_edges, (100, 0, 0),2)

    cv.imshow("Connected Edges", image)



if __name__ == "__main__":
    file = "sc_pres.png"
    image = cv.imread(file)
    drawFocalPointsPipeline(image, lsd(image))

    cv.waitKey(0)
    cv.destroyAllWindows()