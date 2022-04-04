#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
using namespace std;
#pragma once
class Body
{
	float3 pos;//current position
	float3 dir;//current velocity
	float3 acc;//current acceleration

public:
	float mass;
	Body(float3 pos, float3 dir, float mass);
	~Body();
	void showInfo();
	void updateDir(float3 newDir);
	void updateDir(float newX, float newY, float newZ);
	void updatePos(float3 newPos);
	void updatePos(float newX, float newY, float newZ);
	void updateAcc(float3 newAcc);
	void updateAcc(float newX, float newY, float newZ);
	float3 getPos() {
		return pos;
	}
	float3 getDir() {
		return dir;
	}
	float3 getAcc() {
		return acc;
	}
};

