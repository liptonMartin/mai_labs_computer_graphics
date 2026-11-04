#version 450 core

layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_uv;

layout(location=1) out vec3 out_normal;
layout(location=2) out vec3 out_world_pos;
layout(location=3) out vec2 out_uv;

layout(std140, set=0, binding=0) uniform SceneUniforms {
    mat4 view;
    mat4 proj;
    vec3 cameraPos;
    float ambientStrength;
    vec3 ambientColor;
    float _pad0;
    vec3  dir_direction;
    float dir_intensity;
    vec3  dir_color;
    float _pad1;
    uint  pointLightCount;
    uint  spotLightCount;
    vec2  _pad2;
} scene;

layout(std140, set=1, binding=0) uniform ModelUniforms {
    mat4 model;         // offset 0
    float shininess;    // offset 64
    vec3 specularColor; // offset 80 (vec3 выравнивается по 16)
    float _pad0;        // offset 92
} model_data;

void main() {
    const vec4 world_pos = model_data.model * vec4(in_position, 1.0);

    gl_Position   = scene.proj * scene.view * world_pos;
    out_world_pos = world_pos.xyz;
    out_normal    = transpose(inverse(mat3(model_data.model))) * in_normal;
    out_uv        = in_uv;
}
