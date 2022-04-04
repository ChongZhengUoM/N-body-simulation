#pragma once
#include "mySystem.h"

class GPUSystem :
    public mySystem
{
	float3* hostPos, * devPos;
	float3* hostDir, * devDir;
	float3* hostAcc, * devAcc;
	float* hostMass, * devMass;
public:
	GPUSystem(string filename) :mySystem(filename) {
		loadData2Device();
	}
	GPUSystem(string filename, int countN) :mySystem(filename, countN) {
		loadData2Device();
	}
	GPUSystem(unsigned int count, int pd, int pu, int dd, int du, int md, int mu) :mySystem(count, pd, pu, dd, du, md, mu) {
		loadData2Device();
	}
	void loadData2Device();
	void iterate();
};

