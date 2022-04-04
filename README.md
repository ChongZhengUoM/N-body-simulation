# N-body simulation
Author: Chong Zheng  
Email: chong.zheng@postgrad.manchester.ac.uk
## Introduction
This project is to implement the N-body algorithm using CPU and GPU respectively and then apply OpenGL to visualize the results. 

Some professional comparisons on performances between CPU and GPU versions would also be conducted. 

## Main classes' intro
Body
> This class is extracted from each particle.  
> Includes the particle's current position(float3), velocity(float3), acceleration(float3) and mass(float). 

mySystem
> This class represents the system that consists of massive particles.  
> Includes a vector(class Body) and the count(int) of bodies.  
> This class can be instantiated using a csv file or generating some random data. 

CPUSystem(publicly inherit from mySystem)
> Derived from mySystem, It has a new function(iterate) to calculate accelerations based on all particles in the system.  
> After that, this function would update all particles' position and velocity using the accelerations. 
> The whole process runs only on CPU. 

GPUSystem(publicly inherit from mySystem)
> In the construction function of this class, the data would be loaded into device memory by calling a function(loadData2Device).  
> Same as CPUSystem, it has a new function(iterate) to realize the N-body algorithm, but in a different flow.  
> The function(iterate) would call a CUDA kernel function where GPU would take charge of all calculations, including accelerations, velocities and positions. 
> Finally all new data after an iteration would be synchronized to CPU memory.  

Camera
> This class is for movements of the camera view, including moving around and looking around. 

## Requirements
* NVIDIA GeForce GTX 1050 Ti

* Cuda compilation tools, release 11.6, V11.6.112  
Build cuda_11.6.r11.6/compiler.30978841_0  

* GLFW 3.3

## Configuration
* int widthWindow = 800, heightWindow = 600;
* const float sf = 1e-6;//softerning factor
* const float G = 1.0f;//gravitational constant
* const float deltaT = 0.001f;//time granularity

## Usage
Using different classes(GPUSystem or CPUSystem) can implement the algorithm on different devices. 

Also loading different counts of particles can come up with more experimental information.  