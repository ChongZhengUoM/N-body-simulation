#ifndef CAMERA_H
#define CAMERA_H
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class camera
{
private:
	float deltaTime;
	float lastTime;
	
	float sensitivity;
	float lastX, lastY;
	float pitch, yaw;
	bool firstMouse = true;

public:
	glm::vec3 cameraPos;
	glm::vec3 cameraFro;
	glm::vec3 cameraUp;
	camera(float initX=400.0f, float initY = 300.0f, float initPitch = 0.0f, float initYaw = -90.0f, float initSensitivity = 0.05f) {
		cameraPos = glm::vec3(0.0f, 0.0f, 8.0f);
		cameraFro = glm::vec3(0.0f, 0.0f, -1.0f);
		cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
		deltaTime = 0.0f;
		lastTime = 0.0f;
		lastX = initX;
		lastY = initY;
		pitch = initPitch;
		yaw = initYaw;
		sensitivity = initSensitivity;
	}
	void delta(GLFWwindow* window) {
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastTime;
		lastTime = currentFrame;
	}
	void move(GLFWwindow* window, float speed);
	void lookAround(GLFWwindow* window, double xpos, double ypos);
};

#endif // !CAMERA_H

