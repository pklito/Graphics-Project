import numpy as np

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

def segmentIntersection(a1, a2, b1, b2, threshold = 0):
    top = (b2[0] - b1[0]) * (a1[1] - b1[1]) - (b2[1] - b1[1]) * (a1[0] - b1[0])
    utop = (a2[1] - a1[1]) * (b1[0] - a1[0]) - (a2[0] - a1[0]) * (b1[1] - a1[1])
    bottom = (a2[0] - a1[0]) * (b2[1] - b1[1]) - (a2[1] - a1[1]) * (b2[0] - b1[0])
    
    if bottom == 0:
        return None
    t = top / bottom
    u = utop / bottom
    
    a_len = np.linalg.norm(np.array(a2) - a1)
    b_len = np.linalg.norm(np.array(b2) - b1)

    if t < 0 - threshold / a_len or t > 1 + threshold / a_len or u < 0 - threshold / b_len or u > 1 + threshold / b_len:
        return None
    
    return a1 + t * (np.asarray(a2) - a1)