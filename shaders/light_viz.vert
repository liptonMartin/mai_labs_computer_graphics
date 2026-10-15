#version 450 core

layout(location=0) in vec3 in_position;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 color;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(in_position, 1.0);
}