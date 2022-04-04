#include "./include/Body.h"

__device__ float3 float3_sub(float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 float3_add(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 float3_dot(float3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

//acc seems not needed to be passed in!!
__global__ void calc_Acc(float3* devPos, float3* devAcc, float* devMass, int bodyCount, float sf, float G)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < bodyCount) {
		float3 PosI = devPos[idx];
		float3 AccI = make_float3(0.0, 0.0, 0.0);
		float massI = devMass[idx];
		for (int a = 0; a < bodyCount; a++) {
			float3 r = float3_sub(devPos[a], PosI);
			float rMod2 = r.x * r.x + r.y * r.y + r.z * r.z;
			float3 numerator = float3_dot(float3_dot(r, devMass[a]), G);
			//the reciprocal of denominator 
			float denominatorR = pow((pow(rMod2, 2) + pow(sf, 2)), 2 / 3);
			AccI = float3_add(AccI, float3_dot(numerator, denominatorR));
		}
		devAcc[idx] = AccI;
	}
}

__global__ void calc_Update(float3* devPos, float3* devDir, float3* devAcc, int bodyCount, float deltaT)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < bodyCount) {
		float3 PosI = devPos[idx];
		float3 DirI = devDir[idx];
		float3 AccI = devAcc[idx];
		devPos[idx] = float3_add(float3_add(PosI, float3_dot(DirI, deltaT)), float3_dot(AccI, 0.5 * deltaT * deltaT));
		devDir[idx] = float3_add(DirI, float3_dot(AccI, deltaT));
	}
}

extern "C" void cuda_kernel(float3 * devPos, float3 * devDir, float3 * devAcc, float* devMass, int bodyCount, float sf, float G, float deltaT) {
	dim3 blockSize = 1024;
	dim3 gridSize = bodyCount / 1024 + (bodyCount % 1024 != 0);
	calc_Acc <<<gridSize, blockSize >>> (devPos, devAcc, devMass, bodyCount, sf, G);
	cudaDeviceSynchronize();
	calc_Update <<<gridSize, blockSize >>> (devPos, devDir, devAcc, bodyCount, deltaT);
	cudaDeviceSynchronize();
}