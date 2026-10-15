#version 450 core

layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_color;
layout(location=2) in vec3 in_normal;

layout(location=0) out vec3 out_color;
layout(location=1) out vec3 out_normal;
layout(location=2) out vec3 out_world_pos;

layout(std140, set=0, binding=0) uniform SceneUniforms {
    mat4 view;             // offset 0
    mat4 proj;             // offset 64
    vec3 cameraPos;        // offset 128
    float ambientStrength; // offset 140
    vec3 ambientColor;     // offset 144
    float _pad0;           // offset 156
    vec3  dir_direction;   // offset 160
    float dir_intensity;   // offset 172
    vec3  dir_color;       // offset 176
    float _pad1;           // offset 188
} scene;

layout(std140, set=1, binding=0) uniform ModelUniforms {
    mat4 model;         // offset 0
    vec3 albedo;        // offset 64
    float shininess;    // offset 76
    vec3 specularColor; // offset 80
    float _pad0;        // offset 92
} model_data;

void main() {
    const vec4 world_pos = model_data.model * vec4(in_position, 1.0);

    gl_Position   = scene.proj * scene.view * world_pos;
    out_world_pos = world_pos.xyz;
    out_normal    = transpose(inverse(mat3(model_data.model))) * in_normal;

    out_color = in_color * model_data.albedo;
}