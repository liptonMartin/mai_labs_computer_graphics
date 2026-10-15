#version 450 core

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 color;
} pc;

layout(location=0) out vec4 out_color;

void main() {
    out_color = pc.color;
}