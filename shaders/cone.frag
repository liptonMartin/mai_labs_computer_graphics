#version 450 core

layout(location=0) in vec3 in_color;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec3 in_world_pos;

layout(location=0) out vec4 out_color;

void main() {
    const vec3 n = normalize(in_normal);
    out_color = vec4(n * 0.5 + 0.5, 1.0);
}