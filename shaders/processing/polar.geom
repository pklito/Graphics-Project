#version 450

layout (lines) in;
layout (line_strip, max_vertices = 2) out;

in vec2 phiTheta[];

void main() {
    float phi = phiTheta[i].x;
    float theta = phiTheta[i].y;

    vec3 position;
    position.x = sin(theta) * cos(phi);
    position.y = sin(theta) * sin(phi);
    position.z = cos(theta);

    // Emit the vertex
    gl_Position = vec4(position, 1.0);
    EmitVertex();
    EndPrimitive();
}
