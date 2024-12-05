import math
import numpy as np
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import random
from itertools import combinations
def yawpitchrolldecomposition(R):

    sin_x    = math.sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])    
    validity  = sin_x < 1e-6
    if not validity:
        z1    = math.atan2(R[2,0], R[2,1])     # around z1-axis
        x      = math.atan2(sin_x,  R[2,2])     # around x-axis
        z2    = math.atan2(R[0,2], -R[1,2])    # around z2-axis
    else: # gimbal lock
        z1    = 0                                         # around z1-axis
        x      = math.atan2(sin_x,  R[2,2])     # around x-axis
        z2    = 0                                         # around z2-axis

    return np.array([[z1], [x], [z2]])

trans = [[[-1.86, -1.69, -0.9], [-1.83, 0.59, 3.96]], [[-1.81, -1.65, -0.87], [-1.76, -0.14, 3.02]], [[-1.66, 1.65, 0.39], [2.78, -0.63, 4.26]], [[1.81, 1.2, -0.51], [1.02, -0.33, 4.13]], [[-1.86, -1.85, -1.06], [0.86, -0.4, 3.8]], [[0.59, -1.7, -0.6], [0.97, -0.39, 3.94]], [[2.19, 0.02, -0.08], [-1.84, 0.08, 3.12]], [[1.76, 0.78, 1.51], [-2.06, -0.02, 3.85]], [[0.01, 2.65, 1.12], [1.79, -0.53, 3.91]], [[2.1, -0.36, 0.17], [1.8, -0.38, 3.94]]]
rvec = trans[0][0]
print(rvec)
rmat, _ = cv.Rodrigues(np.array(rvec))

yawpitchroll_angles = -180*yawpitchrolldecomposition(rmat)/math.pi
print(yawpitchroll_angles)
yawpitchroll_angles[0,0] = (360-yawpitchroll_angles[0,0])%360 # change rotation sense if needed, comment this line otherwise
yawpitchroll_angles[1,0] = yawpitchroll_angles[1,0]+90

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def get_options(mat):
    vx = mat[:,0]
    vy = mat[:,1]
    vz = mat[:,2]
    flips = [(1,1), (-1,1), (1,-1), (-1,-1)]
    lst = [vx, vy, vz]
    combs = list(combinations(lst, 2))
    combs += [a[::-1] for a in combs]
    
    return [np.array([s1*a, s2*b, np.cross(s1*a,s2*b)]).T for a, b in combs for s1, s2 in flips]

rvec_base = trans[0][0]
mat_base, _ = cv.Rodrigues(np.array(rvec_base))
options = get_options(mat_base)
# for m in options:
#     color = '#' + ''.join([np.random.choice(list('0123456789ABCDEF')) for _ in range(6)])
#     ax.quiver(0,0,0, m[0,0], m[1,0], m[2,0], color='r')
#     ax.quiver(0,0,0, m[0,1], m[1,1], m[2,1], color='g')
#     ax.quiver(0,0,0, m[0,2], m[1,2], m[2,2], color='b')


def draw_tris(ax, mat, r = 'r', g = 'g', b = 'b', offset = (0,0,0), linestyle = ''):
    o = offset
    ax.quiver(o[0], o[1], o[2], mat[0,0] + o[0],mat[1,0] + o[1],mat[2,0] + o[2],color=r, linestyle=linestyle)
    ax.quiver(o[0], o[1], o[2], mat[0,1] + o[0],mat[1,1] + o[1],mat[2,1] + o[2],color=g, linestyle=linestyle)
    ax.quiver(o[0], o[1], o[2], mat[0,2] + o[0],mat[1,2] + o[1],mat[2,2] + o[2],color=b, linestyle=linestyle)

def get_comp(base):
    return lambda m: (np.dot(m[:,0], base[:,0]) + np.dot(m[:,1], base[:,1]) + np.dot(m[:,2], base[:,2]))/3

mats = [sorted(get_options(cv.Rodrigues(np.array(t[0]))[0]),key=get_comp(np.eye(3)))[-1] for t in trans]
total = sum(mats)
total = np.array([total[:,0]/np.linalg.norm(total[:,0]),total[:,1]/np.linalg.norm(total[:,1]),total[:,2]/np.linalg.norm(total[:,2])]).T


total[:,2] = np.cross(total[:,0], total[:,1])
mat_base = total

ctr = 0
valid_mats = []
for t in trans:
    rvec = t[0]
    mat, _ = cv.Rodrigues(np.array(rvec))
    comp = get_comp(mat_base)
    closest_mat = sorted([m for m in get_options(mat)], key=comp )[-1]
    val1, val2, val3 = random(), random(), random()
    print(t[0], comp(closest_mat), ctr)
    ctr += 1
    #draw_tris(ax, mat, color, color, color)
    offset = closest_mat @ np.array([0.0,0.0,0.0])
    if comp(closest_mat) > 0.98:
        valid_mats.append(closest_mat)
        draw_tris(ax, closest_mat, 'b', 'b', 'b', offset = offset)
    else:
        draw_tris(ax, closest_mat, 'y', 'y', 'y', offset = offset)
    
def yawpitchrolldecomposition(R):

    sin_x    = math.sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])    
    validity  = sin_x < 1e-6
    if not validity:
        z1    = math.atan2(R[2,0], R[2,1])     # around z1-axis
        x      = math.atan2(sin_x,  R[2,2])     # around x-axis
        z2    = math.atan2(R[0,2], -R[1,2])    # around z2-axis
    else: # gimbal lock
        z1    = 0                                         # around z1-axis
        x      = math.atan2(sin_x,  R[2,2])     # around x-axis
        z2    = 0                                         # around z2-axis

    return np.array([[z1], [x], [z2]])


print(yawpitchrolldecomposition(mat_base))
print(yawpitchrolldecomposition(np.array([mat_base[:,2], mat_base[:,0], mat_base[:,1]]).T))

print(yawpitchrolldecomposition(np.array([mat_base[:,1], mat_base[:,2], mat_base[:,0]]).T))
print(np.pi,np.pi/2,np.pi/4)
        

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of Rotation Vectors')

plt.show()
