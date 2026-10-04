#version 450 core

layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_color;

layout(location=0) out vec3 out_color;

layout(std140, set=0, binding=0) uniform GlobalUniforms {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 color;
} uniforms;

void main() {
    gl_Position = uniforms.proj * uniforms.view * uniforms.model * vec4(in_position, 1.0);
    out_color = in_color * uniforms.color.rgb;
}