#version 330 core

out vec4 f_color;

uniform sampler2D texture;
uniform int is_x;
const int size = 5;
const int half_size = 3;

// 5x5 Kernel, single dimension (only half because the 1,2, and 4,5 elements are symmetrical)
const float weights[3] = float[3](0.40262, 0.244201, 0.0544884);

void main() {    
    vec3 result = texelFetch(texture, ivec2(gl_FragCoord.xy), 0).rgb * weights[0];
    ivec2 tex_size = textureSize(texture, 0);
    ivec2 coord = ivec2(gl_FragCoord.xy);

    if (is_x == 1) {
        for (int i = 1; i < half_size; ++i) {
            result += texelFetch(texture, ivec2(gl_FragCoord.xy) + ivec2(i, 0), 0).rgb  * weights[i];
            result += texelFetch(texture, ivec2(gl_FragCoord.xy) - ivec2(i, 0), 0).rgb * weights[i];
        }
    } else {
        for (int i = 1; i < half_size; ++i) {
            result += texelFetch(texture, ivec2(gl_FragCoord.xy) + ivec2(0, i), 0).rgb * weights[i];
            result += texelFetch(texture, ivec2(gl_FragCoord.xy) - ivec2(0, i), 0).rgb * weights[i];
        }
    }

    f_color = vec4(result, 1.0);
}
