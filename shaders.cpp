#include "shaders.h"

shaders::shaders(const string vertexPath, const string fragmentPath) {
	string vertexCode, fragmentCode;
	unsigned int vertex, fragment;
	ifstream vertexFile(vertexPath, ios::in);
	ifstream fragmentFile(fragmentPath, ios::in);

	stringstream vertexStream;
	vertexStream << vertexFile.rdbuf();
	vertexCode = vertexStream.str();
	vertex = glCreateShader(GL_VERTEX_SHADER);
	const char* vCode = vertexCode.c_str();
	glShaderSource(vertex, 1, &vCode, NULL);
	glCompileShader(vertex);
	//get error info
	int  success;
	char infoLog[512];
	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success); if (!success) {
		glGetShaderInfoLog(vertex, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	stringstream fragmentStream;
	fragmentStream << fragmentFile.rdbuf();
	fragmentCode = fragmentStream.str();
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	const char* fCode = fragmentCode.c_str();
	glShaderSource(fragment, 1, &fCode, NULL);
	glCompileShader(fragment);
	//get error info
	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success); if (!success) {
		glGetShaderInfoLog(fragment, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	glLinkProgram(ID);

	glDeleteShader(vertex);
	glDeleteShader(fragment);
}

void shaders::use() {
	glUseProgram(ID);
}