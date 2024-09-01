import json
import glm
import math
import os

def parse_vec3(vec_str):
    # Remove 'vec3(' and ')' and split by ','
    values = vec_str.replace('vec3(', '').replace(')', '').split(',')
    return glm.vec3(float(values[0]), float(values[1]), float(values[2]))

def parse_vec4(vec_str):
    # Remove 'vec4(' and ')' and split by ','
    values = vec_str.replace('vec4(', '').replace(')', '').split(',')
    return glm.vec4(float(values[0]), float(values[1]), float(values[2]), float(values[3]))

def parse_mat4(m_view_str):
    # Split the string by lines and then by spaces to get the matrix values
    rows = m_view_str.split('\n')
    matrix = []
    for row in rows:
        values = row.replace('[', '').replace(']', '').split()
        matrix.append([float(v) for v in values])
    return glm.transpose(glm.mat4(*matrix))

# Open the JSON file
with open('debug2.json', 'r+', encoding='utf-16') as file:
    try:
        # Load the JSON data
        data = json.load(file)
    except json.JSONDecodeError:
        # Replace the first line with "{"data":["
        file.seek(0)
        file.write('{"data":[                                    ')

        # Replace the last line with "]}"
        file.seek(0, os.SEEK_END)
        file.write('{}]}')
        # Load the JSON data
        file.seek(0)
        data = json.load(file)

# Access the data from the "data" key
data_value = data.get('data')

# Convert the 100th entry
entry = data_value[100]
print(entry)
yaw = float(entry['yaw'])
pitch = float(entry['pitch'])
m_view = parse_mat4(entry['m_view'])
up = parse_vec3(entry['up'])
right = parse_vec3(entry['right'])
forward = parse_vec3(entry['forward'])
x_inf = parse_vec4(entry['x inf '])
proj = parse_mat4(entry['proj'])

# Print the converted values
print(f"Yaw: {yaw}")
print(f"Pitch: {pitch}")
print(f"m_view: {glm.mat4(*[[round(value, 3) for value in row] for row in m_view])}")
print(f"Up: {up}")
print(f"Right: {right}")
print(f"Forward: {forward}")
print(f"x_inf: {x_inf}")
print(f"proj: {glm.mat4(*[[round(value, 3) for value in row] for row in proj])}")

yaw_rad = math.radians(yaw)
pitch_rad = math.radians(pitch)

print(f"cos(yaw): {glm.cos(yaw_rad)}")
print(f"sin(yaw): {glm.sin(yaw_rad)}")
print(f"cos(pitch): {glm.cos(pitch_rad)}")
print(f"sin(pitch): {glm.sin(pitch_rad)}")

print(f"Forward: {glm.vec3(glm.cos(yaw_rad) * glm.cos(pitch_rad), glm.sin(pitch_rad), glm.sin(yaw_rad) * glm.cos(pitch_rad))}")
print(f"Right: {glm.vec3(glm.cos(yaw_rad + glm.pi() / 2), 0, glm.sin(yaw_rad + glm.pi() / 2))}")
print(f"Up: {glm.vec3(-glm.cos(yaw_rad) * glm.sin(pitch_rad), glm.cos(pitch_rad), -glm.sin(yaw_rad) * glm.sin(pitch_rad))}")

print(m_view * glm.vec4(10000,0,0,1))