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

int evaluate(int bodyNum, int iterateNum) {
    CPUSystem Cms("./resource/data/c_0000.csv", bodyNum);
    GPUSystem Gms("./resource/data/c_0000.csv", bodyNum);
    //unit sec
    LARGE_INTEGER t1, t2, tc;
    QueryPerformanceFrequency(&tc);

    //CPU
    QueryPerformanceCounter(&t1);
    Cms.iterate(iterateNum);
    QueryPerformanceCounter(&t2);
    double t_CPU = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
    
    //GPU_1 -> transfer data to GPU
    QueryPerformanceCounter(&t1);
    Gms.loadData2Device();
    QueryPerformanceCounter(&t2);
    double t_GPU_1 = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
    //GPU_2 -> iterations
    QueryPerformanceCounter(&t1);
    Gms.iterate(iterateNum);
    QueryPerformanceCounter(&t2);
    double t_GPU_2 = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
    //GPU_3 -> transfer data back to CPU
    QueryPerformanceCounter(&t1);
    Gms.synchr();
    QueryPerformanceCounter(&t2);
    double t_GPU_3 = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;

    //report
    cout << "[+]ACCURACY:" << Cms.matchUp(Gms.allBodies) << endl;
    cout << " CPU runtime:" << t_CPU << "sec" << endl;
    cout << " CPU runtime:" << t_GPU_1 << "sec\t" << t_GPU_2 << "sec\t" << t_GPU_3 << "sec" << endl;

    cout << "[+]Speedup without data transfer:" << t_CPU/t_GPU_2 << endl;
    cout << "[+]Speedup with data transfer:" << t_CPU/(t_GPU_1+t_GPU_2+t_GPU_3) << endl;
    return 0;
}

/*
int visualize(int bodyNum) {
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
    glBufferData(GL_ARRAY_BUFFER, bodyNum * sizeof(float3), Gms.hostPos, GL_STREAM_DRAW);

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

    double lastTime = glfwGetTime();
    int nbFrames = 0;

    GPUSystem Gms("./resource/data/c_0000.csv", bodyNum);

    // //visualization
    while (!glfwWindowShouldClose(window)) {

        // Measure speed
        double currentTime = glfwGetTime();
        nbFrames++;
        if (currentTime - lastTime >= 1.0) { // If last prinf() was more than 1 sec ago
            // printf and reset timer
            printf("FPS:\t%d\n", nbFrames);
            nbFrames = 0;
            lastTime += 1.0;
        }

        //clean buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //reload delta value
        ourCamera.delta(window);

        localInput(window);

        //size of points
        glPointSize(5.0f);
        //frame
        ourShaders.use();
        glBindVertexArray(VAO);
        for (body item:Gms.getbody()) {
            glm::mat4 model, view, projection;
            view = glm::lookat(ourcamera.camerapos, ourcamera.camerapos + ourcamera.camerafro, ourcamera.cameraup);
            projection = glm::perspective(glm::radians(fov), (float)(widthwindow / heightwindow), 0.1f, 100.0f);

            model = glm::translate(model, glm::vec3(item.getpos().x, item.getpos().y, item.getpos().z));

            gluniformmatrix4fv(glgetuniformlocation(ourshaders.id, "modelt"), 1, gl_false, glm::value_ptr(model));
            gluniformmatrix4fv(glgetuniformlocation(ourshaders.id, "viewt"), 1, gl_false, glm::value_ptr(view));
            gluniformmatrix4fv(glgetuniformlocation(ourshaders.id, "projt"), 1, gl_false, glm::value_ptr(projection));

            gldrawarrays(gl_points, 0, 1);
        }

        //swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        //calculate once
        Gms.iterate(1);
        Gms.synchr();
    }

    glfwTerminate();
	return 0;
}
*/

int main() {
    evaluate(2000,20);
    //visualize(2000);
    return 0;
}