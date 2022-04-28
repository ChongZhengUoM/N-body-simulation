#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <shaders.h>
#include <camera.h>
#include <string>
#include "./include/Body.h"
#include "./include/GPUSystem.h"
#include "./include/CPUSystem.h"

//global setting
int widthWindow = 800, heightWindow = 600;
std::string nameWindow("N-body");
float fov=45.0f;

//register camera
camera ourCamera(widthWindow / 2.0, heightWindow / 2.0);

void localLookAround(GLFWwindow* window, double xpos, double ypos) {
    ourCamera.lookAround(window, xpos, ypos);
}

void localInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    ourCamera.move(window, 2.5f);
}

void localScroll(GLFWwindow* window, double xoffset, double yoffset) {
    if (fov >= 1.0f && fov <= 45.0f)
        fov -= yoffset;
    if (fov <= 1.0f)
        fov = 1.0f;
    if (fov >= 45.0f)
        fov = 45.0f;
}

int evaluate() {
    //initial our solar system with CPU and 500 particles
    //CPUSystem Cms("./resource/data/c_0000_500.csv");
    //initial our solar system with GPU and 500 particles
    //GPUSystem Gms("./resource/data/c_0000_500.csv");
    //initial our solar system with CPU and 2000 particles
    CPUSystem Cms("./resource/data/c_0000_2000.csv");
    //initial our solar system with GPU and 2000 particles
    GPUSystem Gms("./resource/data/c_0000_2000.csv");

    //check the correctness of the GPU version
    for (int i = 0; i < 20; i++) {
        Cms.iterate();
        Gms.iterate();
        cout << Cms.matchUp(Gms.allBodies) << endl;
    }
    return 0;
}

int visualize() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    //create a window
    GLFWwindow* window = glfwCreateWindow(widthWindow, heightWindow, nameWindow.c_str(), NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    //glad
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //window setting
    glViewport(0, 0, widthWindow, heightWindow);

    //shaders
    shaders ourShaders("./shader.vs", "./shader.fs");

    //vertical locations
    float vertices[] = {
        0.0f, 0.0f, 0.0f,
    };

    //VAO
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    //VBO
    unsigned int VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    //VBO
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    //Z buffer
    glEnable(GL_DEPTH_TEST);

    //mouse callback
    glfwSetCursorPosCallback(window, localLookAround);
    //wheel callback
    glfwSetScrollCallback(window, localScroll);

    //hide cursor
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    GPUSystem Gms("./resource/data/c_0000_500.csv");

    double lastTime = glfwGetTime();
    int nbFrames = 0;

    // //visualization
    while (!glfwWindowShouldClose(window)) {

        // Measure speed
        double currentTime = glfwGetTime();
        nbFrames++;
        if (currentTime - lastTime >= 1.0) { // If last prinf() was more than 1 sec ago
            // printf and reset timer
            printf("%f ms/frame\n", 1000.0 / double(nbFrames));
            nbFrames = 0;
            lastTime += 1.0;
        }

        //clean buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //reload delta value
        ourCamera.delta(window);

        localInput(window);

        //size of points
        glPointSize(1.0f);
        //frame
        ourShaders.use();
        glBindVertexArray(VAO);
        for (Body item:Gms.getBody()) {
            glm::mat4 model, view, projection;
            view = glm::lookAt(ourCamera.cameraPos, ourCamera.cameraPos + ourCamera.cameraFro, ourCamera.cameraUp);
            projection = glm::perspective(glm::radians(fov), (float)(widthWindow / heightWindow), 0.1f, 100.0f);

            model = glm::translate(model, glm::vec3(item.getPos().x, item.getPos().y, item.getPos().z));

            glUniformMatrix4fv(glGetUniformLocation(ourShaders.ID, "modelT"), 1, GL_FALSE, glm::value_ptr(model));
            glUniformMatrix4fv(glGetUniformLocation(ourShaders.ID, "viewT"), 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(glGetUniformLocation(ourShaders.ID, "projT"), 1, GL_FALSE, glm::value_ptr(projection));

            glDrawArrays(GL_POINTS, 0, 1);
        }

        //swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        //calculate once
        Gms.iterate();
    }

    glfwTerminate();
	return 0;
}

int main() {
    //evaluate();
    visualize();
    return 0;
}