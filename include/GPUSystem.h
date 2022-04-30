#pragma once
#include <windows.h>
#include "mySystem.h"

class GPUSystem :
    public mySystem
{
	float3* hostDir, * devDir;
	float3* hostAcc, * devAcc;
	float* hostMass, * devMass;
public:
	float3* hostPos, * devPos;
	GPUSystem(string filename) :mySystem(filename) {
	}
	GPUSystem(string filename, int countN) :mySystem(filename, countN) {
	}
	GPUSystem(unsigned int count, int pd, int pu, int dd, int du, int md, int mu) :mySystem(count, pd, pu, dd, du, md, mu) {
	}
	void loadData2Device();
	void iterate(int iterNum);
	void synchr();
};

