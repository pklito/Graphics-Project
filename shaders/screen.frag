#version 330 core

uniform sampler2D Texture;

layout (location = 0) out vec4 out_color;

void main() {
    ivec2 at = ivec2(gl_FragCoord.xy);
    //at.y = textureSize(Texture, 0).y - at.y - 1;
    out_color = texelFetch(Texture, at, 0);
}