#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;
#pragma once
class shaders
{
public:
	unsigned int ID;
	shaders(const string vertexPath, const string fragmentPath);
	void use();
};

#endif