import cv2
import numpy as np

# Load the image
image = cv2.imread('test3.png', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny edge detector
edges = cv2.Canny(blurred, 5, 10)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
