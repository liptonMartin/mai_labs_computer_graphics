#version 450 core

layout(location=0) in vec3 in_color;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec3 in_world_pos;

layout(location=0) out vec4 out_color;

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
    mat4 model;
    vec3 albedo;
    float shininess;
    vec3 specularColor;
    float _pad0;
} model_data;

struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
    float radius;
};

struct SpotLight {
    vec3 position;
    float _pad0;
    vec3 direction;
    float intensity;
    vec3 color;
    float innerAngle;
    float outerAngle;
    float radius;
    float _pad1;
    float _pad2;
};

layout(std430, set=0, binding=1) readonly buffer PointLights {
    PointLight lights[];
} pointLights;

layout(std430, set=0, binding=2) readonly buffer SpotLights {
    SpotLight lights[];
} spotLights;

void main() {
    const vec3 N = normalize(in_normal);
    const vec3 V = normalize(scene.cameraPos - in_world_pos);

    const vec3 ambient = scene.ambientStrength * scene.ambientColor * model_data.albedo;

    const vec3 L = normalize(-scene.dir_direction);
    const vec3 H = normalize(L + V);

    const float NdotL = max(dot(N, L), 0.0);
    const float NdotH = max(dot(N, H), 0.0);

    const float spec_factor = (NdotL > 0.0) ? pow(NdotH, max(model_data.shininess, 1e-3)) : 0.0;

    const vec3 diffuse  = model_data.albedo * NdotL;
    const vec3 specular = model_data.specularColor * spec_factor;

    const vec3 directional =
        (diffuse + specular) * scene.dir_color * scene.dir_intensity;

    const vec3 final_color = ambient + directional;

    vec3 dummy = vec3(0.0);
    if (scene.pointLightCount > 0u) dummy += pointLights.lights[0].color;
    if (scene.spotLightCount  > 0u) dummy += spotLights.lights[0].color;

    out_color = vec4(final_color + dummy * 0.0, 1.0);
}