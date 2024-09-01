import re

# Read the content of the debug.txt file
with open('debug.txt', 'r',encoding='utf-16') as file:
    data = file.read()

# Normalize the whitespace to ensure consistent matching]
# Regex to match the vec4 lines
pattern = re.compile(r'vec4\(\s*([-+]?\d*\.\d+|\d+)\s*,\s*([-+]?\d*\.\d+|\d+)\s*,\s*([-+]?\d*\.\d+|\d+)\s*,\s*([-+]?\d*\.\d+|\d+)\s*\)')

# Find all matches
matches = re.findall(pattern, data)
# Group the matches into frames (3 vectors per frame)
frames = [[tuple(map(float, match)) for match in matches[i:i+3]] for i in range(0, len(matches), 3)]

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot each frame's vectors
for i, frame in enumerate(frames):
    # Extracting X, Y, Z components of each vector
    x = [vec[0]/100000 for vec in frame]
    y = [vec[1]/100000 for vec in frame]
    z = [vec[2]/100000 for vec in frame]

    # Plotting the vectors
    ax.scatter(x[0], y[0], z[0], color='r', label=f'Frame {i+1} Vector 1' if i == 0 else "")
    ax.scatter(x[1], y[1], z[1], color='g', label=f'Frame {i+1} Vector 2' if i == 0 else "")
    ax.scatter(x[2], y[2], z[2], color='b', label=f'Frame {i+1} Vector 3' if i == 0 else "")

ax.scatter()
# Setting labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adding a legend
ax.legend()

# Display the plot
plt.show()
