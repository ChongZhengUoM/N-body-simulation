#include "./include/CPUSystem.h"

float3 _float3_sub(float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
float3 _float3_add(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
float3 _float3_dot(float3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

void CPUSystem::iterate() {
	//loadAcc
	for (auto& itemI : allBodies) {
		float3 acc = { 0.0,0.0,0.0 };
		for (auto itemJ : allBodies) {
			float3 r = _float3_sub(itemJ.getPos(), itemI.getPos());
			float rMod2 = r.x * r.x + r.y * r.y + r.z * r.z;
			float3 numerator = _float3_dot(_float3_dot(r, itemJ.mass), G);
			//the reciprocal of denominator 
			float denominatorR = pow((pow(rMod2, 2) + pow(sf, 2)), 2 / 3);
			acc = _float3_add(acc, _float3_dot(numerator, denominatorR));
		}
		itemI.updateAcc(acc);
	}
	//updateData
	for (auto& itemI : allBodies) {
		float3 newPos = _float3_add(_float3_add(itemI.getPos(), _float3_dot(itemI.getDir(), deltaT)), _float3_dot(itemI.getAcc(), 0.5 * deltaT * deltaT));
		float3 newDir = _float3_add(itemI.getDir(), _float3_dot(itemI.getAcc(), deltaT));
		itemI.updatePos(newPos);
		itemI.updateDir(newDir);
	}
}