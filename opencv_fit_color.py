import cv2 as cv
import numpy as np
from opencv import lsd
import matplotlib.pyplot as plt
from util import *

HEIGHT = 400
WIDTH = 600
def toRange(v, min, max, newmin, newmax):
    if max == min:
        return (v-min)*(newmax-newmin) + newmin
    return (v - min)*(newmax - newmin)/(max-min)+newmin

def loss_function(point, rho, phi):
    return pow(point[1]*np.sin((phi)) + point[0]*np.cos((phi)) - rho,2)

def min_loss(points, lines):
    """
    Assumes lines are list of [rho, phi] where rho is offset and phi is radian angle
    Assumes that the points are actually inverted? like (y, x) instead of (x, y)
    opencv is stupid
    """
    return sum([min([loss_function(point, rho, phi) for point in points]) for rho, phi in lines]) / len(lines)
    
def sum_loss(phi, theta, lines):
    return min_loss(get_focal_points(phi, theta), lines)

def regress_lines(lines, iterations = 500, refinement_iterations = 100, refinement_area=0.3):

    best_loss = 100000
    best_phi_theta = (0, 0)
    for i in range(0,iterations):
        phi = np.random.rand()*np.pi
        theta = np.random.rand()*np.pi/2
        if sum_loss(phi, theta, lines) < best_loss:
            best_loss = sum_loss(phi, theta, lines)
            best_phi_theta = (phi, theta)

    current_phi, current_theta = best_phi_theta
    for i in range(0,refinement_iterations):
        phi = np.random.rand()*refinement_area + current_phi - refinement_area/2
        theta = np.random.rand()*refinement_area + current_theta - refinement_area/2
        if sum_loss(phi, theta, lines) < best_loss:
            best_loss = sum_loss(phi, theta, lines)
            best_phi_theta = (phi, theta)
    print(" The loss is ", best_loss)
    return best_phi_theta, best_loss

def which_line(focal_points, line, threshold = 700):
    x_loss, y_loss, z_loss = loss_function(focal_points[0], *line), loss_function(focal_points[1], *line), loss_function(focal_points[2], *line)
    if min([x_loss, y_loss, z_loss]) >= threshold:
        return None
    
    if x_loss <= y_loss and x_loss <= z_loss:
        return "x"
    elif y_loss <= x_loss and y_loss <= z_loss:
        return "y"
    else:
        return "z"
    
def which_color(line_type):
    if line_type == None:
        return (0,0,0)
    if line_type == "x":
        return (0, 0, 100)
    elif line_type == "y":
        return (0, 100, 0)
    else:
        return (100, 20, 20)

def get_camera_angles(image, iterations = 1000, method = 'hough', refinement_iterations = 500):
    if method == 'hough':
        # Flip the image along the x and y axis
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        def get_edges_image(image, blur = (5, 5),thresh1 = 5, thresh2 = 10):
            blurred = cv.GaussianBlur(image, blur, 0)
            return cv.Canny(blurred, thresh1, thresh2)
        
        canny = get_edges_image(gray)

        #image = cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)])
        lines = cv.HoughLines(canny, 1, np.pi / 180, 50, None, 0, 0)
        lines = lines[:,0,:]
    elif method == "lsd":
        
        lines = lsd(image)
        lines = [(np.sign(np.arctan2(b[0] - a[0], b[1] - a[1]))*(a[1]*b[0]-b[1]*a[0])/np.linalg.norm(b-a), np.fmod(-np.arctan2(b[0] - a[0], b[1] - a[1]) + np.pi,np.pi)) for a, b in lines if np.linalg.norm(b-a) > 10]
        lines = np.array(lines)
    else:
        return None
    
    return regress_lines(lines, iterations=iterations, refinement_iterations=refinement_iterations)[0]

def get_focal_points(phi, theta):
    sc_points = [(1/(np.tan(theta)*np.sin(phi)), 1/np.tan(phi)),
            (0,        - np.tan(phi)),
            (-np.tan(theta)/np.sin(phi),     1/np.tan(phi))]
    
    # Its (1/ASPECT_RATIO * width/2 * p[0] + width/2, height/2 * p[1] + height) so height is used in both multiplications
    return [np.array([HEIGHT/2 * p[0] + WIDTH/2,  HEIGHT/2 * p[1] + HEIGHT/2]) for p in sc_points]

def get_focal_points_projection(phi, theta):
    # estimated method, use get_focal_points(phi,theta) for the analytical, more accurate solution
    # I used this function to debug my lookAt matrix and camera matrix
    lookat_matrix = getCameraTransformationMatrix(phi, theta) # lookAt
    camera_matrix = getIntrinsicsMatrix()      # camera intrinsics (the function names are bad im sorry)
    return [camera_matrix @ vec3ToEuclidian((lookat_matrix @ np.array([1000000,0,0,1]))), camera_matrix @ vec3ToEuclidian((lookat_matrix @ np.array([0,1000000,0,1]))),camera_matrix @ vec3ToEuclidian((lookat_matrix @ np.array([0,0,1000000,1])))]

def show_points_on_image(image, points, lines, cam_phi, cam_theta):
    print("focal points: ", points)
    
    boundary = min(200, max(max([abs(point[0] - WIDTH/2) for point in points]) - WIDTH/2, max([abs(point[1] - HEIGHT/2) for point in points])- HEIGHT/2, 0))
    boundary = int(boundary)
    def convert_coords(point):
        return (int(point[0] + boundary), int(point[1] + boundary))
    
    image = cv.copyMakeBorder(image, boundary, boundary, boundary, boundary, cv.BORDER_CONSTANT, value=(50, 50, 50))
    for rho, phi in lines:
        line_type = which_line(points, (rho, phi))
        color = which_color(line_type)
        a = np.cos(phi)
        b = np.sin(phi)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image, convert_coords((x1, y1)), convert_coords((x2, y2)), color, 2)

    cv.circle(image, convert_coords(points[0]), 10, (0, 0, 255), -1)
    cv.circle(image, convert_coords(points[1]), 10, (0, 255, 0), -1)
    cv.circle(image, convert_coords(points[2]), 10, (200, 0, 0), -1)

    cv.circle(image, convert_coords(points[0]), 3, (255, 255, 255), -1)
    cv.circle(image, convert_coords(points[1]), 3, (255, 255, 255), -1)
    cv.circle(image, convert_coords(points[2]), 3, (255, 255, 255), -1)

    # lookat_matrix = getCameraTransformationMatrix(cam_phi, cam_theta)
    # camera_matrix = getIntrinsicsMatrix()
    # focal_points = get_focal_points(cam_phi, cam_theta)

    # projected_focal_points = [camera_matrix @ vec3ToEuclidian((lookat_matrix @ np.array([1000000,0,0,1]))), camera_matrix @ vec3ToEuclidian((lookat_matrix @ np.array([0,1000000,0,1]))),camera_matrix @ vec3ToEuclidian((lookat_matrix @ np.array([0,0,1000000,1])))]
    # projected_focal_points = [[int(p[0]), int(p[1])] for p in projected_focal_points]
    # focal_points = [[int(p[0]), int(p[1])] for p in focal_points] 
    
    # cv.circle(image, convert_coords(projected_focal_points[0]), 15, (0, 0, 200), -1)
    # cv.circle(image, convert_coords(projected_focal_points[1]), 15, (0, 200, 0), -1)
    # cv.circle(image, convert_coords(projected_focal_points[2]), 15, (200, 0, 0), -1)


    scale_percent = 80  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    cv.imshow('Hough Lines', image)


def draw_vanishing_points_plots(lines, phi, theta, show = True):    
    plt.scatter(lines[:,1], lines[:,0])
    plt.xlabel("phi")
    plt.ylabel("rho")
    plt.title("Hough lines in polar coordinates (rho phi)")
    
    def draw_point(point, color = 'r'):
        points = np.array([(point[1]*np.sin(np.deg2rad(phi)) + point[0]*np.cos(np.deg2rad(phi)), np.deg2rad(phi)) for phi in range(180)])
        points = np.array([p for p in points if np.abs(p[0]) < np.linalg.norm(np.array([WIDTH,HEIGHT]))])
        if len(points) == 0:
            return
        plt.plot(points[:, 1], points[:, 0], color)
    
    fc = get_focal_points(phi, theta)
    draw_point(fc[0], color='r')
    draw_point(fc[1], color='g')
    draw_point(fc[2], color='b')
    if show:
        plt.show()

def draw_vanishing_waves(image, phi, theta):
    # Load the image
    # Flip the image along the x and y axis
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def get_edges_image(image, blur = (5, 5),thresh1 = 5, thresh2 = 10):
        blurred = cv.GaussianBlur(image, blur, 0)
        return cv.Canny(blurred, thresh1, thresh2)
    
    canny = get_edges_image(gray)

    #image = cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)])
    lines = cv.HoughLines(canny, 1, np.pi / 180, 50, None, 0, 0)

    lines = np.array(lines)
    lines = lines[:,0,:]

    lines2 = lsd(image)
    lines2 = [(np.sign(np.arctan2(b[0] - a[0], b[1] - a[1]))*(a[1]*b[0]-b[1]*a[0])/np.linalg.norm(b-a), np.fmod(-np.arctan2(b[0] - a[0], b[1] - a[1]) + np.pi,np.pi)) for a, b in lines2 if np.linalg.norm(b-a) > 10]
    lines2 = np.array(lines2)
    plt.scatter(lines2[:,1], lines2[:,0], color='r')

    print("drawing loss:", sum_loss(phi, theta, lines2))
    
    plt.scatter(lines[:,1], lines[:,0])
    plt.xlabel("phi")
    plt.ylabel("rho")
    plt.title("Hough lines in polar coordinates (rho phi)")

    # plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    
    def draw_point(point, color = 'r'):
        points = np.array([(point[1]*np.sin(np.deg2rad(phi)) + point[0]*np.cos(np.deg2rad(phi)), np.deg2rad(phi)) for phi in range(180)])
        points = np.array([p for p in points if np.abs(p[0]) < np.linalg.norm(np.array([WIDTH,HEIGHT]))])
        if len(points) == 0:
            return
        plt.plot(points[:, 1], points[:, 0], color)
    
    fc = get_focal_points(phi, theta)
    draw_point(fc[0], color='r')
    draw_point(fc[1], color='g')
    draw_point(fc[2], color='b')

    show_points_on_image(image, fc, lines2, phi, theta)
    #cv.waitKey(0)
    plt.show()

if __name__ == "__main__":
    file = 'generated_images/demo_rgb.png'
    image = cv.imread(file, cv.IMREAD_COLOR)
    loss = get_camera_angles(image, iterations = 500, method="lsd")
    draw_vanishing_waves(image, *loss)