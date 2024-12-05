#version 330 core

layout (location = 0) out vec4 fragColor;

in vec2 uv_0;
in vec3 normal;
in vec3 fragPos;

struct Light {
    vec3 position;
    vec3 Ia;
    vec3 Id;
    vec3 Is;
};

uniform Light light;
uniform sampler2D u_texture_0;
uniform vec3 camPos;


void main() {
    float gamma = 2.2;
    vec3 color = texture(u_texture_0, uv_0).rgb;
    if (color.r < 0.1 && color.g < 0.1 && color.b < 0.1) {
        discard;
    }
    if(false || color.r == -1.0){
        color = normal + light.Ia + light.position + light.Id + light.Is + camPos;//dead code gets erased and compile errors happen
    }
    fragColor = vec4(color, 1.0);
}










