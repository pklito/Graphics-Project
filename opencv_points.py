import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import combinations

# points = [[-0.11, 1.17, 3.1], [-0.11, 1.17, 3.1],
# [-3.76, 0.4, 5.61], [-3.76, 0.4, 5.61],
# [-1.15, -0.82, 6.24], [-1.15, -0.82, 6.24],
# [0.13, -0.66, 5.48], [0.13, -0.66, 5.48],
# [-2.66, -0.28, 5.07], [-2.66, -0.28, 5.07],
# [-1.4, -3.95, 9.99], [-1.4, -3.95, 9.99],
# [0.2, -0.67, 7.01], [0.2, -0.67, 7.01],
# [-2.42, -0.14, 4.38], [-2.42, -0.14, 4.38],
# [2.26, 0.21, 6.11], [2.26, 0.21, 6.11],
# [2.26, 0.16, 6.08], [2.26, 0.16, 6.08],
# [-4.55, 0.6, 6.34], [-4.55, 0.6, 6.34]]


def get_options(mat):
    vx = mat[:,0]
    vy = mat[:,1]
    vz = mat[:,2]
    flips = [(1,1), (-1,1), (1,-1), (-1,-1)]
    lst = [vx, vy, vz]
    combs = list(combinations(lst, 2))
    combs += [a[::-1] for a in combs]
    
    return [np.array([s1*a, s2*b, np.cross(s1*a,s2*b)]).T for a, b in combs for s1, s2 in flips]

def get_comp(base):
    return lambda m: (np.dot(m[:,0], base[:,0]) + np.dot(m[:,1], base[:,1]) + np.dot(m[:,2], base[:,2]))/3

def orient_up(mat):
    def tilt(vec):
        return abs(np.dot(vec,np.array([1,0,0])))
    vx = mat[:,0]
    vy = mat[:,1]
    vz = mat[:,2]
    if tilt(vy) < tilt(vx) and tilt(vy) < tilt(vz):
        return mat
    if tilt(vx) < tilt(vz):
        return np.array([vy, vz, vx]).T
    return np.array([vz, vx, vy]).T

def alignTrans(trans, threshold = 0.97, stop_early_percent = 0.8):
    """
    Threshold, similarity that allows two transformations to be grouped together
    stop_early_percent. if % of matrices are similar, return this group without going through all options.
    """
    # Convert rodrigues vectors to matrices
    mats = [(cv.Rodrigues(np.array(t[0]))[0], t[1]) for t in trans]
    # Align matrices to (1, 0, 0), (0, 1, 0), (0, 0, 1)
    mats = [(sorted(get_options(m[0]),key=get_comp(np.eye(3)))[-1], m[1]) for m in mats]
    
    # Get some common matrix
    average_mat = sum([m[0] for m in mats])
    x_temp = average_mat[:,0]/np.linalg.norm(average_mat[:,0])
    y_temp = average_mat[:,1]/np.linalg.norm(average_mat[:,1])
    average_mat = np.array([x_temp, y_temp, np.cross(x_temp, y_temp)]).T

    max_list = []
    not_in = []
    max_amount = 0
    # Overly complicated
    for i in range(len(mats)):
        mat_size = len(mats)
        average_mat = orient_up(average_mat)
        new_mats = [(sorted(get_options(t[0]),key=get_comp(average_mat))[-1], t[1]) for t in mats]
        filtered_mats = [(m[0], m[1]) for m in new_mats if get_comp(average_mat)(m[0]) > threshold]
        if len(filtered_mats) >= stop_early_percent * mat_size:
            max_list = filtered_mats
            not_in = [(m[0], m[1]) for m in new_mats if get_comp(average_mat)(m[0]) <= threshold]
            max_amount = len(new_mats)
            break
        print("Failed to find good average_matrix, ", len(filtered_mats) / mat_size)
        if len(filtered_mats) > max_amount:
            max_list = new_mats
            not_in = [(m[0], m[1]) for m in new_mats if get_comp(average_mat)(m[0]) <= threshold]
            max_amount = len(new_mats)
        average_mat = mats[i][0]

    return max_list, not_in

def matsToCubesWithCamera(mats, camera_mat):
    cam_inv = np.linalg.inv(camera_mat)
    return [(cam_inv @ m[1]).ravel() for m in mats]

def alignCubes(points):
    
    average_fract = [np.average([i - np.floor(i) for i in points[:,0]]), np.average([i - np.floor(i) for i in points[:,1]]),np.average([i - np.floor(i) for i in points[:,2]])]
    print("average fractions:", average_fract)
    # Define the vertices of the cube centered at (1, 2, 1) with side length 2
    points = [[i+(0.5 - j) for i, j in zip(p,average_fract)] for p in points]
    return points

def alignCubesStochastic(points, iterations = 100):
    best_fracts = [0, 0, 0]
    best_losses = [100000, 100000, 100000]


    for iter in range(iterations):
        average_fract = [0,0,0]
        for axis in range(3):
            average_fract[axis] = np.random.rand()-0.5
            new_points = [p[axis] + average_fract[axis] for p in points]
            loss = sum([pow(p - np.floor(p) - 0.5, 2) for p in new_points])/len(new_points)
            if loss < best_losses[axis]:
                best_losses[axis] = loss
                best_fracts[axis] = average_fract[axis]
    print("best fractions:", best_fracts)
    print("best losses:", best_losses)
    return [[i+j for i, j in zip(p,best_fracts)] for p in points]


def matsToCubes(mats):
    """
    Rotate all the cubes around the origin based on the rotation matrices of each,
    and average the cubes so that they fall in one block
    """
    average_mat = sum([m[0] for m in mats])
    x_temp = average_mat[:,0]/np.linalg.norm(average_mat[:,0])
    y_temp = average_mat[:,1]/np.linalg.norm(average_mat[:,1])
    average_mat = np.array([x_temp, y_temp, np.cross(x_temp, y_temp)]).T

    average_mat = sorted(sorted(get_options(average_mat),key=lambda x: -x[2][2])[0:4],key=lambda x: x[0][0])[0]

    points = [t[1] for t in mats]
    points = [(average_mat.T @ np.array(p)).ravel() for p in points]

    points = np.array(points)

    # Cubing
    aligned_points = alignCubesStochastic(points)

    points = np.array(aligned_points)
    return points

def plot_cubes(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([0],[0],[0],c='b')
    screen_points = [[point[0]/point[2], point[1]/point[2],1] for point in points]
    screen_points = np.array(screen_points)
    
    ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')

    def draw_cube(ax, point):
        corner = [np.floor(i) for i in point]
        corners = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ]
        corners = [np.array(p) + np.array(corner) for p in corners]
        cube_vertices = np.array(corners)


        # Define the edges that connect the vertices
        cube_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # Draw the cube
        for edge in cube_edges:
            start, end = edge
            ax.plot(
                [cube_vertices[start][0], cube_vertices[end][0]],
                [cube_vertices[start][1], cube_vertices[end][1]],
                [cube_vertices[start][2], cube_vertices[end][2]],
                color='k'
            )

    for p in points:
        draw_cube(ax,p)

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(screen_points[:,0],screen_points[:,1], c='g')
    # plt.ion()
    plt.show()
    plt.pause(.001)
if __name__ == "__main__":

    trans = [[[-1.86, -1.69, -0.9], [-1.83, 0.59, 3.96]], [[-1.81, -1.65, -0.87], [-1.76, -0.14, 3.02]], [[-1.66, 1.65, 0.39], [2.78, -0.63, 4.26]], [[1.81, 1.2, -0.51], [1.02, -0.33, 4.13]], [[-1.86, -1.85, -1.06], [0.86, -0.4, 3.8]], [[0.59, -1.7, -0.6], [0.97, -0.39, 3.94]], [[2.19, 0.02, -0.08], [-1.84, 0.08, 3.12]], [[1.76, 0.78, 1.51], [-2.06, -0.02, 3.85]], [[0.01, 2.65, 1.12], [1.79, -0.53, 3.91]], [[2.1, -0.36, 0.17], [1.8, -0.38, 3.94]]]
    mats, excluded = alignTrans(trans)
    points = matsToCubes(mats)
    plot_cubes(points)
    
    