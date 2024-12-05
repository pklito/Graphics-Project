#version 330 core

uniform sampler2D Texture;
uniform int flip_x = 0;
uniform int flip_y = 0;

layout (location = 0) out vec4 out_color;

void main() {
    ivec2 at = ivec2(gl_FragCoord.xy);
    if(flip_y == 1) {
        at.y = textureSize(Texture, 0).y - at.y - 1;
    }
    if(flip_x == 1) {
        at.x = textureSize(Texture, 0).x - at.x - 1;
    }
    out_color = texelFetch(Texture, at, 0);
}