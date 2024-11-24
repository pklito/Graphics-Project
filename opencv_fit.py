import cv2 as cv
import numpy as np

def toRange(v, min, max, newmin, newmax):
    if max == min:
        return (v-min)*(newmax-newmin) + newmin
    return (v - min)*(newmax - newmin)/(max-min)+newmin

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
        theta = 2*np.random.rand()*np.pi - np.pi
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

def get_focal_points(phi, theta):
    sc_points = [(1/(np.tan(theta)*np.sin(phi)), 1/np.tan(phi)),
            (0,        - np.tan(phi)),
            (np.tan(theta)/np.sin(phi),     1/np.tan(phi))]
    
    return [np.array([300 * p[0] + 300,  200 * p[1] + 200]) for p in sc_points]

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
    
    plt.scatter(lines[:,0,1], lines[:,0,0])
    plt.xlabel("phi")
    plt.ylabel("rho")
    plt.title("Hough lines in polar coordinates (rho phi)")

    # plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    
    def draw_point(point, color = 'r'):
        points = np.array([(point[0]*np.sin(np.deg2rad(phi)) + point[1]*np.cos(np.deg2rad(phi)), np.deg2rad(phi)) for phi in range(180)])
        points = np.array([p for p in points if np.abs(p[0]) < np.linalg.norm(np.array([600,400]))])
        plt.plot(points[:, 1], points[:, 0], color)
    
    fc = get_focal_points(phi, theta)
    draw_point(fc[0], color='r')
    draw_point(fc[1], color='g')
    draw_point(fc[2], color='b')

    plt.show()

if __name__ == "__main__":
    print(get_camera_angles('sc_rgb.png', iterations = 100))
    draw_vanishing_waves('sc_rgb.png', 0.9277301344864572, 2.009863182359205)
