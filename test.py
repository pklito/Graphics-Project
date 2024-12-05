import cv2 as cv
import numpy as np


# Load the image
image = cv.imread('sc_white_2.png', cv.IMREAD_COLOR)
# Flip the image along the x and y axis
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blue_channel, green_channel, red_channel = cv.split(image)
# Apply GaussianBlur to reduce noise and improve contour detection

def get_edges_image(image, blur = (5, 5),thresh1 = 5, thresh2 = 10):
    blurred = cv.GaussianBlur(image, blur, 0)
    return cv.Canny(blurred, thresh1, thresh2)

canny_split = cv.cvtColor(cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)]),cv.COLOR_BGR2GRAY)

canny = get_edges_image(gray)

#image = cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)])
lines = cv.HoughLines(canny, 1, np.pi / 180, 50, None, 0, 0)


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
        cv.line(overlay, pt1, pt2, (0,0,255), linewidth, cv.LINE_AA)

print("we got", len(lines), "lines")
if lines is None:
    exit(1)
def toRange(v, min, max, newmin, newmax):
    if max == min:
        return (v-min)*(newmax-newmin) + newmin
    return (v - min)*(newmax - newmin)/(max-min)+newmin

#drawHoughLines(image, lines, 1)

max_rho, min_rho, max_theta, min_theta = np.sqrt(canny.shape[0]**2 + canny.shape[1]**2), -np.sqrt(canny.shape[0]**2 + canny.shape[1]**2), np.pi, 0
print(max_rho,min_rho,max_theta,min_theta)
print(max(lines[:,:,0]),min(lines[:,:,0]),max(lines[:,:,1]),min(lines[:,:,1]))
print(canny.shape)

def angToScreen(rho,theta):
    return (int(toRange(theta,min_theta,max_theta,0,canny.shape[1])), int(toRange(rho,min_rho,max_rho,0,canny.shape[0])))
for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 100*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1800*(-b)), int(y0 - 1000*(a)))
    #cv.line(image, pt1, pt2, (0,0,255,50), 1, cv.LINE_AA)
    cv.circle(image,(int(toRange(theta,min_theta,max_theta,0,canny.shape[1])), int(toRange(rho,min_rho,max_rho,0,canny.shape[0]))), 5, (0,0,255,255))

cv.line(image,angToScreen(0,0),angToScreen(0,np.pi),(255,0,0,255),thickness=2)

def draw_point(image, point):
    point = point[1], point[0]
    points = np.array([angToScreen(point[0]*np.sin(np.deg2rad(phi)) + point[1]*np.cos(np.deg2rad(phi)), np.deg2rad(phi)) for phi in range(180)], dtype=np.int32)
    cv.polylines(image, [points], False, (0, 0, 0))




actual_cam_matrix = np.array([[    -0.493335 ,  1.49012e-07 ,      0.86984 ,      2.80745 ],
[     0.533611 ,     0.789727 ,     0.302641 ,      3.04372 ],
[    -0.686935 ,     0.613459 ,      -0.3896 ,       -12.46 ],
[            0 ,            0 ,            0 ,            1 ]])
cam_theta = np.arcsin(actual_cam_matrix[0][0])
cam_phi = np.arccos(actual_cam_matrix[1][1])
screen_width = canny.shape[1]
screen_height = canny.shape[0]
print("actual", cam_theta, cam_phi)

# for i in range(0,100):
#     frame = image.copy()
#     phi = cam_phi - i/1000
#     theta = cam_theta
#     draw_point(frame, (toRange(np.cos(phi)*np.tan(theta),-1,1,0,screen_width), toRange(-np.tan(phi),-1,1,0,screen_height)))
#     draw_point(frame, (toRange(0,-1,1,0,screen_width), toRange(np.cos(phi),-1,1,0,screen_height)))
#     draw_point(frame, (toRange(-np.cos(phi)/np.tan(theta),-1,1,0,screen_width), toRange(-np.tan(phi),-1,1,0,screen_height)))
#     cv.imshow('Contours', frame)
#     cv.waitKey(50)
# print(theta, phi)

def draw_phi_theta(image, phi, theta):
    draw_point(image, (toRange(np.cos(phi)*np.tan(theta),-1,1,0,screen_width), toRange(-np.tan(phi),-1,1,0,screen_height)))
    draw_point(image, (toRange(0,-1,1,0,screen_width), toRange(np.cos(phi),-1,1,0,screen_height)))
    draw_point(image, (toRange(-np.cos(phi)/np.tan(theta),-1,1,0,screen_width), toRange(-np.tan(phi),-1,1,0,screen_height)))

def regression(lines):
    lines = lines[:,0,:]
    def loss(point, lines):
        return sum([pow(point[0]*np.sin((phi)) + point[1]*np.cos((phi)) - rho,2) for rho, phi in lines])
    
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

    phi = cam_phi
    theta = cam_theta
    print("stam", sum_loss(0, 0, lines))
    print("sum", sum_loss(phi, theta, lines))

    best_loss = 100000
    best_phi_theta = (0, 0)
    for i in range(0,2000):
        phi = np.random.rand()*np.pi
        theta = np.random.rand()*np.pi
        if sum_loss(phi, theta, lines) < best_loss:
            best_loss = sum_loss(phi, theta, lines)
            best_phi_theta = (phi, theta)
    print(best_loss,best_phi_theta)
    draw_phi_theta(image, best_phi_theta[0], best_phi_theta[1])

    pass

regression(lines)

# Display the result
cv.imshow('Contours', image)
cv.waitKey(0)
cv.destroyAllWindows()
