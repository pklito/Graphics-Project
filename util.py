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

def edgeDistance(edge1, edge2):
    p1, p2 = edge1
    p3, p4 = edge2
    def dist(point, line):
        return np.linalg.norm(np.array(point) - getEdgeProjection(point, line))
    return min(dist(p1, edge2), dist(p2, edge2), dist(p3, edge1), dist(p4, edge1))

def lineMatrixToPairs(lines):
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
    origin, end = p1, p4
    if np.dot(p3 - p1, avg1) < 0:
        origin, end = p3, p2
    new_length = np.dot(end - origin, avg1) / (np.dot(avg1, avg1))
    return np.array([origin, origin + new_length * avg1])

def combineParallelLines(lines, max_distance = 5, max_angle = 3):
    new_lines = []
    cancel = False
    for i in range(len(lines) - 1):
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