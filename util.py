import numpy as np
WIDTH = 600
HEIGHT = 400
def toRange(v, min, max, newmin, newmax):
            if max == min:
                return (v-min)*(newmax-newmin) + newmin
            return (v - min)*(newmax - newmin)/(max-min)+newmin

def polarToLine(rho, theta, width = WIDTH, height = HEIGHT):
    """
    Used for drawing polar lines on screen
    Converts vector to line to two points, which are off screen."""
    max_rho, min_rho, max_theta, min_theta = np.sqrt(width*width + height*height), -np.sqrt(width*width + height*height), np.pi, 0
    if rho < min_rho or rho > max_rho or theta < min_theta or theta > max_theta:
        return None, None
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1800*(-b)), int(y0 + 1800*(a)))
    pt2 = (int(x0 - 1800*(-b)), int(y0 - 1800*(a)))
    return pt1, pt2
    
def clamp(val, v_min, v_max):
    return min(v_max, max(v_min,val))

def getEdgeProjection(config, edge):
    p1 = np.asarray(edge[0])
    p2 = np.asarray(edge[1])
    point = np.asarray(config)

    # Get a vector of the given path edge
    edge_vector = p2 - p1
    edge_length_squared = np.dot(edge_vector,edge_vector)
    if edge_length_squared <= 0.001:    
        return p2, 1

    # Vector from path start to current point
    point_vector = point - p1

    # T is the fraction along the path the projection is on.
    t_distance = edge_vector.dot(point_vector)
    t = t_distance / edge_length_squared

    projection = None
    if(t < 0):
        projection = p1
    elif (t>1):
        projection = p2
    else:
        projection = t*edge_vector + p1
    return projection

def lineIntersection(a1, a2, b1, b2):
    xdiff = (a1[0] - a2[0], b1[0] - b2[0])
    ydiff = (a1[1] - a2[1], b1[1] - b2[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(a1, a2), det(b1, b2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def pointSegmentDistance(point, a1, a2):
    u = ((point[0] - a1[0]) * (a2[0] - a1[0]) + (point[1] - a1[1]) * (a2[1] - a1[1])) / np.linalg.norm(np.array(a2) - a1)
    if u < 0:
        return np.linalg.norm(np.array(point) - np.array(a1))
    if u > 1:
        return np.linalg.norm(np.array(point) - np.array(a2))
    return np.linalg.norm(np.array(a1 + u * (np.array(a2)-a1)) - point)

def segmentDistance(segment1, segment2):
    a1, a2 = segment1
    b1, b2 = segment2
    intersection, t, u, a_len, b_len = _segmentIntersection(a1, a2, b1, b2)
    if intersection is not None:
        return 0
    return min(pointSegmentDistance(a1, b1, b2), pointSegmentDistance(a2, b1, b2), pointSegmentDistance(b1, a1, a2), pointSegmentDistance(b2, a1, a2))

def _segmentIntersection(a1, a2, b1, b2, threshold = 0):
    top = (b2[0] - b1[0]) * (a1[1] - b1[1]) - (b2[1] - b1[1]) * (a1[0] - b1[0])
    utop = (a2[1] - a1[1]) * (b1[0] - a1[0]) - (a2[0] - a1[0]) * (b1[1] - a1[1])
    bottom = (a2[0] - a1[0]) * (b2[1] - b1[1]) - (a2[1] - a1[1]) * (b2[0] - b1[0])
    
    if bottom == 0:
        return None, None, None, None, None
    t = top / bottom
    u = utop / bottom
    
    a_len = np.linalg.norm(np.array(a2) - a1)
    b_len = np.linalg.norm(np.array(b2) - b1)
    if threshold is not None and (t < 0 - threshold / a_len or t > 1 + threshold / a_len or u < 0 - threshold / b_len or u > 1 + threshold / b_len):
        return None, t, u, a_len, b_len
    
    return a1 + t * (np.asarray(a2) - a1), t, u, a_len, b_len

def edgeDistance(edge1, edge2):
    p1, p2 = edge1
    p3, p4 = edge2
    def dist(point, line):
        return np.linalg.norm(np.array(point) - getEdgeProjection(point, line))
    return min(dist(p1, edge2), dist(p2, edge2), dist(p3, edge1), dist(p4, edge1))

def lineMatrixToPairs(lines):
    if np.array(lines[0]).shape != (1, 4):
        print("Error: lines must be a matrix of shape (1, 4)", np.array(lines[0]).shape)
        return lines
    return [(np.array(line[0][0:2]), np.array(line[0][2:4])) for line in lines]

def combineEdges(line1, line2):
    """
    Takes two parallel edges and combines them into a single edge.
    """
    # Find the two points that are the farthest apart, to get the longest continuous edge.
    p1, p2 = line1
    p3, p4 = line2
    p1 ,p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
    # The new edge may not be parallel to the previous edges.
    # We will create one that is more parallel to the previous edges.
    edge1 = p2 - p1
    edge2 = p4 - p3
    # Make both edges face the same direction (not be 180 degrees off)
    if np.dot(edge1,edge2) < 0:
        edge2 = -edge2
        ptemp = p3
        p3 = p4
        p4 = ptemp
    
    avg1 = edge1 + edge2
    origin = p1
    if np.dot(p3 - origin, avg1) < 0:
        origin = p3
    new_length = max(np.dot(p2 - origin, avg1) / (np.dot(avg1, avg1)), np.dot(p4 - origin, avg1) / (np.dot(avg1, avg1)))
    return np.array([origin, origin + new_length * avg1])

def combineParallelLines(lines, max_distance = 5, max_angle = 3):
    new_lines = []
    cancel = False
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if np.abs(np.dot(lines[i][1] - lines[i][0], lines[j][1] - lines[j][0])) / (np.linalg.norm(lines[i][1] - lines[i][0]) * np.linalg.norm(lines[j][1] - lines[j][0])) > np.cos(np.radians(max_angle)):
                if edgeDistance(lines[i], lines[j]) < max_distance:
                    new_lines = new_lines + [combineEdges(lines[i], lines[j])] + lines[i+1:j] + lines[j+1:]
                    cancel = True
            
            if cancel:
                break
        if cancel:
            break        
        new_lines.append(lines[i])

    if cancel:
        return combineParallelLines(new_lines)
    return new_lines


def pointInConvexPolygon(point, polygon):
    def get_side(p1, edge):
        return (p1[0] - edge[0][0]) * (edge[1][1] - edge[0][1]) - (p1[1] - edge[0][1]) * (edge[1][0] - edge[0][0])
    sign = get_side(point, [polygon[0], polygon[1]])
    if sign == 0:
        return True
    return all(0 <= sign * get_side(point, [polygon[i], polygon[i+1]]) for i in range(1, len(polygon) - 1))

def faceCircumference(face):
    return sum(np.linalg.norm(np.array(face[i])-face[i+1]) for i in range(-1,len(face) - 1))

def toEuclidian(vec4):
    return np.array(vec4[0:3])/vec4[2]

def getIntrinsicsMatrix(focal_length=1, width = WIDTH, height = HEIGHT):
    focal_length_x = HEIGHT / 2
    focal_length_y = HEIGHT / 2
    camera_matrix = np.array([
        [focal_length_x, 0, WIDTH/2],
        [0, focal_length_y, HEIGHT/2],
        [0, 0, 1]
    ])
    return camera_matrix

def getCameraTransformationMatrix(pitch, yaw):
    # pitch 0: looking straight up,
    # pitch pi/2: looking straight ahead
    pitch = np.pi/2-pitch
    yaw = yaw + np.pi/2
    matrix = np.array([[np.sin(yaw), 0, -np.cos(yaw), 0],
                       [-np.cos(yaw)*np.sin(pitch), np.cos(pitch), -np.sin(yaw)*np.sin(pitch), 0],
                       [-np.cos(yaw)*np.cos(pitch), -np.sin(pitch), -np.sin(yaw)*np.cos(pitch), 0],
                       [0, 0, 0, 1]])
    return matrix
if __name__ == "__main__":
    p = [0.5, 0.5]
    poly = [(0,0), (0,1), (1,1), (1,0)]
    print(pointInConvexPolygon(p, poly))
    print(faceCircumference(poly))