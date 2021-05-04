#version 330 core

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 color;

uniform mat4 V;
uniform mat4 P;
uniform mat4 M;

out vec3 fragmentColor;

void main()
{
    gl_Position = P * V * M * vec4(pos, 1);
    fragmentColor = color;
}
