#include "./include/GPUSystem.h"

extern "C" void cuda_kernel(float3 * devPos, float3 * devDir, float3 * devAcc, float* devMass, int bodyCount, float sf, float G, float deltaT);

void GPUSystem::loadData2Device() {
	hostPos = new float3[bodyCount];
	hostDir = new float3[bodyCount];
	hostAcc = new float3[bodyCount];
	hostMass = new float[bodyCount];
	cudaMalloc((void**)&devPos, bodyCount * sizeof(float3));
	cudaMalloc((void**)&devDir, bodyCount * sizeof(float3));
	cudaMalloc((void**)&devAcc, bodyCount * sizeof(float3));
	cudaMalloc((void**)&devMass, bodyCount * sizeof(float));
	for (int i = 0; i < bodyCount; i++) {
		hostPos[i] = allBodies[i].getPos();
		hostDir[i] = allBodies[i].getDir();
		hostAcc[i] = allBodies[i].getAcc();
		hostMass[i] = allBodies[i].mass;
	}
	cudaMemcpy(devPos, hostPos, bodyCount * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(devDir, hostDir, bodyCount * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(devAcc, hostAcc, bodyCount * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(devMass, hostMass, bodyCount * sizeof(float), cudaMemcpyHostToDevice);
}

void GPUSystem::iterate() {
	cuda_kernel(devPos, devDir, devAcc, devMass, bodyCount, sf, G, deltaT);
	cudaMemcpy(hostPos, devPos, bodyCount * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostDir, devDir, bodyCount * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostAcc, devAcc, bodyCount * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostMass, devMass, bodyCount * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < bodyCount; i++) {
		allBodies[i].updatePos(hostPos[i]);
		allBodies[i].updateDir(hostDir[i]);
		allBodies[i].updateAcc(hostAcc[i]);
		allBodies[i].mass = hostMass[i];
	}
}
