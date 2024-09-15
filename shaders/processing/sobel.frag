#version 330 core

uniform sampler2D Texture;

layout (location = 0) out float out_color;

void main() {
    float sobel_x[9] = float[9](
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    );

    float sobel_y[9] = float[9](
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1
    );

    vec3 gx = vec3(0.0);
    vec3 gy = vec3(0.0);

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            ivec2 at = ivec2(gl_FragCoord.xy);

            vec3 sample = texelFetch(Texture, ivec2(i + at.x, j + at.y), 0).rgb;
            gx += sample * sobel_x[3*(i+1) + j+1];
            gy += sample * sobel_y[3*(i+1) + j+1];
        }
    }

    float gradient_magnitude = length(gx) + length(gy); //should be sqrt(gx^2 + gy^2 cdot (1,1,1))
    out_color = gradient_magnitude;
}
