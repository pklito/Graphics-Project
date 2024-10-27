import cv2 as cv
import numpy as np


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


def toRange(v, min, max, newmin, newmax):
    if max == min:
        return (v-min)*(newmax-newmin) + newmin
    return (v - min)*(newmax - newmin)/(max-min)+newmin

def angToScreen(image, rho,theta, max_rho, min_rho, max_theta, min_theta):
    return (int(toRange(theta,min_theta,max_theta,0,image.shape[1])), int(toRange(rho,min_rho,max_rho,0,image.shape[0])))
    

def regress_lines(lines, screen_width, screen_height, iterations = 500):
    lines = lines[:,0,:]
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
        theta = np.random.rand()*np.pi
        if sum_loss(phi, theta, lines) < best_loss:
            best_loss = sum_loss(phi, theta, lines)
            best_phi_theta = (phi, theta)

    print(" The loss is ", best_loss)
    return best_phi_theta

def get_camera_angles(file, iterations = 500):
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
    return regress_lines(lines, image.shape[1], image.shape[0], iterations=iterations)


if __name__ == "__main__":
    print(get_camera_angles('sc_rgb.png', iterations = 1000))
