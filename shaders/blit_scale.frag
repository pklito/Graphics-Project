#version 330 core

uniform sampler2D Texture;
uniform int flip_x = 0;
uniform int flip_y = 0;

uniform int outputWidth = 600;
uniform int outputHeight = 400;


layout (location = 0) out vec4 out_color;

ivec2 convertCoord(ivec2 coord) {
    return ivec2(textureSize(Texture,0).x*coord.x/outputWidth, textureSize(Texture,0).y*coord.y/outputHeight);
}   

void main() {
    ivec2 at = ivec2(gl_FragCoord.xy);
    if(flip_y == 1) {
        at.y = outputHeight - at.y - 1;
    }
    if(flip_x == 1) {
        at.x = outputWidth - at.x - 1;
    }
    vec4 color_sum = vec4(0.0);
    int count = 0;
    for(int i = convertCoord(at).x; i <= convertCoord(at + ivec2(1,0)).x ; i++) {
        for(int j = convertCoord(at).y; j <= convertCoord(at + ivec2(1,0)).y; j++) {
            color_sum += texelFetch(Texture, ivec2(i, j), 0);
            count += 1;
        }
    }
    color_sum /= float(count);
    out_color = color_sum;
}