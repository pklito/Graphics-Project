#version 330 core

uniform sampler2D Texture;

layout (location = 0) out vec4 out_color;

void main() {
    ivec2 at = ivec2(gl_FragCoord.xy);
    //at.y = textureSize(Texture, 0).y - at.y - 1;
    float value = texelFetch(Texture, at, 0).r;
    if(value < 0.05){
        discard;
    }
    out_color = vec4(value,value,value,min(1,2*value));
}