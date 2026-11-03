#version 450 core

layout(location=1) in vec3 in_normal;
layout(location=2) in vec3 in_world_pos;
layout(location=3) in vec2 in_uv;

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

layout(set=1, binding=1) uniform sampler2D albedo_tex;

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
    const vec3 albedo = texture(albedo_tex, in_uv).rgb;
    const vec3 N = normalize(in_normal);
    const vec3 V = normalize(scene.cameraPos - in_world_pos);

    const vec3 ambient = scene.ambientStrength * scene.ambientColor * albedo;

    const vec3 Ld = normalize(-scene.dir_direction);
    const vec3 Hd = normalize(Ld + V);
    const float NdotLd = max(dot(N, Ld), 0.0);
    const float NdotHd = max(dot(N, Hd), 0.0);
    const float spec_d = (NdotLd > 0.0) ? pow(NdotHd, max(model_data.shininess, 1e-3)) : 0.0;
    const vec3 directional =
        (albedo * NdotLd + model_data.specularColor * spec_d)
        * scene.dir_color * scene.dir_intensity;

    vec3 point_accum = vec3(0.0);
    const uint pl_count = min(scene.pointLightCount, uint(pointLights.lights.length()));
    for (uint i = 0u; i < pl_count; ++i) {
        const PointLight pl = pointLights.lights[i];

        const vec3  to_light = pl.position - in_world_pos;
        const float dist2    = max(dot(to_light, to_light), 1e-6);
        const float dist     = sqrt(dist2);
        const vec3  L        = to_light / dist;

        const float attenuation = 1.0 / dist2;

        const vec3  H        = normalize(L + V);
        const float NdotL    = max(dot(N, L), 0.0);
        const float NdotH    = max(dot(N, H), 0.0);
        const float spec_p   = (NdotL > 0.0) ? pow(NdotH, max(model_data.shininess, 1e-3)) : 0.0;

        float radius_falloff = 1.0;
        if (pl.radius > 0.0) {
            const float t = clamp(1.0 - (dist / pl.radius), 0.0, 1.0);
            radius_falloff = t * t;
        }

        const vec3 contribution =
            (albedo * NdotL + model_data.specularColor * spec_p)
            * pl.color * pl.intensity * attenuation * radius_falloff;

        point_accum += contribution;
    }

    vec3 spot_accum = vec3(0.0);
    const uint sl_count = min(scene.spotLightCount, uint(spotLights.lights.length()));
    for (uint i = 0u; i < sl_count; ++i) {
        const SpotLight sl = spotLights.lights[i];

        const vec3  to_light = sl.position - in_world_pos;
        const float dist2    = max(dot(to_light, to_light), 1e-6);
        const float dist     = sqrt(dist2);
        const vec3  L        = to_light / dist;

        const vec3 spotDir = normalize(sl.direction);

        const float theta = dot(-L, spotDir);

        const float cos_inner = cos(sl.innerAngle);
        const float cos_outer = cos(sl.outerAngle);

        const float epsilon = max(cos_inner - cos_outer, 1e-4);
        float spotAttenuation = clamp((theta - cos_outer) / epsilon, 0.0, 1.0);

        const float attenuation = 1.0 / dist2;

        const vec3  H      = normalize(L + V);
        const float NdotL  = max(dot(N, L), 0.0);
        const float NdotH  = max(dot(N, H), 0.0);
        const float spec_s = (NdotL > 0.0) ? pow(NdotH, max(model_data.shininess, 1e-3)) : 0.0;

        float radius_falloff = 1.0;
        if (sl.radius > 0.0) {
            const float t = clamp(1.0 - (dist / sl.radius), 0.0, 1.0);
            radius_falloff = t * t;
        }

        const vec3 contribution =
            (albedo * NdotL + model_data.specularColor * spec_s)
            * sl.color * sl.intensity * attenuation
            * spotAttenuation * radius_falloff;

        spot_accum += contribution;
    }

    const vec3 final_color = ambient + directional + point_accum + spot_accum;

    out_color = vec4(final_color, 1.0);
}
