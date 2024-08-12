#version 330 core

uniform sampler2D Texture;

layout (location = 0) out float out_color;

void main() {

    out_color = gradient_magnitude;
}
