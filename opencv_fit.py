import cv2 as cv
import numpy as np
from opencv import lsd

def toRange(v, min, max, newmin, newmax):
    if max == min:
        return (v-min)*(newmax-newmin) + newmin
    return (v - min)*(newmax - newmin)/(max-min)+newmin

def regress_lines(lines, screen_width, screen_height, iterations = 500):

    def min_loss(points, lines):
        """
        Assumes lines are list of [rho, phi] where rho is offset and phi is radian angle
        Assumes that the points are actually inverted? like (y, x) instead of (x, y)
        opencv is stupid
        """
        return sum([min([pow(point[1]*np.sin((phi)) + point[0]*np.cos((phi)) - rho,2) for point in points]) for rho, phi in lines]) / len(lines)
        
    def sum_loss(phi, theta, lines):
        return min_loss(
            [(toRange(np.cos(phi)*np.tan(theta),-1,1,0,screen_width), toRange(-np.tan(phi),-1,1,0,screen_height)),
            (toRange(0,-1,1,0,screen_width), toRange(np.cos(phi),-1,1,0,screen_height)), 
            (toRange(-np.cos(phi)/np.tan(theta),-1,1,0,screen_width), toRange(-np.tan(phi),-1,1,0,screen_height))]
            , lines)

    best_loss = 100000
    best_phi_theta = (0, 0)
    for i in range(0,iterations):
        phi = np.random.rand()*np.pi
        theta = 2*np.random.rand()*np.pi - np.pi
        if sum_loss(phi, theta, lines) < best_loss:
            best_loss = sum_loss(phi, theta, lines)
            best_phi_theta = (phi, theta)

    print(" The loss is ", best_loss)
    return best_phi_theta

def get_camera_angles(file, iterations = 500, method = 'hough'):
    if method == 'hough':
        # Load the image
        image = cv.imread(file, cv.IMREAD_COLOR)
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
        
        image = cv.imread(file, cv.IMREAD_COLOR)
        lines = lsd(image)
        lines = [(np.sign(np.arctan2(b[0] - a[0], b[1] - a[1]))*(a[1]*b[0]-b[1]*a[0])/np.linalg.norm(b-a), np.fmod(-np.arctan2(b[0] - a[0], b[1] - a[1]) + np.pi,np.pi)) for a, b in lines if np.linalg.norm(b-a) > 10]
        lines = np.array(lines)
    else:
        return None
    
    return regress_lines(lines, image.shape[1], image.shape[0], iterations=iterations)

def get_focal_points(phi, theta):
    sc_points = [(1/(np.tan(theta)*np.sin(phi)), 1/np.tan(phi)),
            (0,        - np.tan(phi)),
            (-np.tan(theta)/np.sin(phi),     1/np.tan(phi))]
    
    return [np.array([200 * p[0] + 300,  200 * p[1] + 200]) for p in sc_points]

def show_points_on_image(image, points, lines):
    print("focal points: ", points)
    
    boundary = 350
    def convert_coords(point):
        return (int(point[0] + boundary), int(point[1] + boundary))
    
    image = cv.copyMakeBorder(image, boundary, boundary, boundary, boundary, cv.BORDER_CONSTANT, value=(255, 0, 0))
    for rho, phi in lines:
        a = np.cos(phi)
        b = np.sin(phi)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image, convert_coords((x1, y1)), convert_coords((x2, y2)), (0, 0, 55), 2)

    cv.circle(image, convert_coords(points[0]), 10, (0, 0, 255), -1)
    cv.circle(image, convert_coords(points[1]), 10, (0, 255, 0), -1)
    cv.circle(image, convert_coords(points[2]), 10, (200, 0, 0), -1)

    scale_percent = 80  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    cv.imshow('Hough Lines', image)

def draw_vanishing_waves(file, phi, theta):
    # Load the image
    image = cv.imread(file, cv.IMREAD_COLOR)
    # Flip the image along the x and y axis
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def get_edges_image(image, blur = (5, 5),thresh1 = 5, thresh2 = 10):
        blurred = cv.GaussianBlur(image, blur, 0)
        return cv.Canny(blurred, thresh1, thresh2)
    
    canny = get_edges_image(gray)

    #image = cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)])
    lines = cv.HoughLines(canny, 1, np.pi / 180, 50, None, 0, 0)
    import matplotlib.pyplot as plt
    lines = np.array(lines)

    lines2 = lsd(image)
    lines2 = [(np.sign(np.arctan2(b[0] - a[0], b[1] - a[1]))*(a[1]*b[0]-b[1]*a[0])/np.linalg.norm(b-a), np.fmod(-np.arctan2(b[0] - a[0], b[1] - a[1]) + np.pi,np.pi)) for a, b in lines2 if np.linalg.norm(b-a) > 10]
    lines2 = np.array(lines2)
    plt.scatter(lines2[:,1], lines2[:,0], color='r')

    plt.scatter(lines[:,0,1], lines[:,0,0])
    plt.xlabel("phi")
    plt.ylabel("rho")
    plt.title("Hough lines in polar coordinates (rho phi)")

    # plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    
    def draw_point(point, color = 'r'):
        points = np.array([(point[0]*np.sin(np.deg2rad(phi)) + point[1]*np.cos(np.deg2rad(phi)), np.deg2rad(phi)) for phi in range(180)])
        points = np.array([p for p in points if np.abs(p[0]) < np.linalg.norm(np.array([600,400]))])
        if len(points) == 0:
            return
        plt.plot(points[:, 1], points[:, 0], color)
    
    fc = get_focal_points(phi, theta)
    draw_point(fc[0], color='r')
    draw_point(fc[1], color='g')
    draw_point(fc[2], color='b')

    show_points_on_image(image, fc, lines2)
    cv.waitKey(0)
    plt.show()

if __name__ == "__main__":
    #print(get_camera_angles('sc_pres_angles.png', iterations = 10000, method="lsd"))
    draw_vanishing_waves('sc_scarce.png', -0.932610459142018, 0.7743767313761)
    print("actual:", np.rad2deg(-0.932610459142018), np.rad2deg(0.7743767313761))
    print("main", np.rad2deg(0.5326743767313761), np.rad2deg(-0.21432610459142018 ))
