import cv2 as cv
import numpy as np

# Load the image
image = cv.imread('test4.png', cv.IMREAD_COLOR)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blue_channel, green_channel, red_channel = cv.split(image)
# Apply GaussianBlur to reduce noise and improve contour detection

def get_edges_image(image, blur = (5, 5),thresh1 = 5, thresh2 = 10):
    blurred = cv.GaussianBlur(image, blur, 0)
    return cv.Canny(blurred, thresh1, thresh2)

canny_split = cv.cvtColor(cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)]),cv.COLOR_BGR2GRAY)
canny = get_edges_image(gray)

canny = np.ones_like(gray)
#image = cv.merge([get_edges_image(blue_channel),get_edges_image(red_channel),get_edges_image(green_channel)])
lines = cv.HoughLines(canny, 1, np.pi / 180, 10, None, 0, 0)

if lines is not None:
    def toRange(v, min, max, newmin, newmax):
        if max == min:
            return (v-min)*(newmax-newmin) + newmin
        return (v - min)*(newmax - newmin)/(max-min)+newmin
    
    max_rho, min_rho, max_theta, min_theta = np.sqrt(canny.shape[0]**2 + canny.shape[1]**2), -np.sqrt(canny.shape[0]**2 + canny.shape[1]**2), np.pi, 0
    print(max_rho,min_rho,max_theta,min_theta)
    print(max(lines[:,:,0]),min(lines[:,:,0]),max(lines[:,:,1]),min(lines[:,:,1]))
    print(canny.shape)
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
        cv.circle(image,(int(toRange(theta,min_theta,max_theta,0,canny.shape[1])), int(toRange(rho,min_rho,max_rho,0,canny.shape[0]))), 2, (255,255,0,255))
    cv.line(image,(0,int(toRange(0,min_rho,max_rho,0,canny.shape[0]))),(canny.shape[1],int(toRange(0,min_rho,max_rho,0,canny.shape[0]))),(255,0,0,255),thickness=2)
    cv.line(image,(0,int(toRange(canny.shape[0],min_rho,max_rho,0,canny.shape[0]))),(canny.shape[1],int(toRange(canny.shape[0],min_rho,max_rho,0,canny.shape[0]))),(0,0,255,255),thickness=2)
    cv.line(image,(0,int(toRange(-canny.shape[0],min_rho,max_rho,0,canny.shape[0]))),(canny.shape[1],int(toRange(-canny.shape[0],min_rho,max_rho,0,canny.shape[0]))),(0,0,255,255),thickness=2)



# Display the result
cv.imshow('Contours', image)
cv.waitKey(0)
cv.destroyAllWindows()
