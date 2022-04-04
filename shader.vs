#version 330 core

layout (location = 0) in vec3 aPos;

uniform mat4 modelT;
uniform mat4 viewT;
uniform mat4 projT;

void main()
{
    gl_Position = projT * viewT * modelT * vec4(aPos, 1.0);
}