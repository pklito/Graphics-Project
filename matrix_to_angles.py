import numpy as np
input = """[    -0.212689 ][ -4.47035e-08 ][      0.97712 ][     0.229041 ]
[      0.49622 ][     0.861452 ][     0.108012 ][     -5.87466 ]
[    -0.841742 ][     0.507839 ][    -0.183222 ][      5.52289 ]
[            0 ][            0 ][            0 ][            1 ]"""
input = input.split("\n")
input = [i.split("][") for i in input]
input = [[float(j.strip().replace('[', '').replace(']', '')) for j in i] for i in input]
actual_cam_matrix = np.array(input)
cam_theta = np.arcsin(actual_cam_matrix[0][0])
cam_phi = np.arccos(actual_cam_matrix[1][1])

print(cam_phi, cam_theta)