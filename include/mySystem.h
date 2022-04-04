#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include <sstream>
using namespace std;
#include "Body.h"

const float sf = 1e-6;//softerning factor
const float G = 1.0f;//gravitational constant
const float deltaT = 0.001f;//time granularity

#pragma once
class mySystem
{
public:
	vector<Body> allBodies;
	unsigned int bodyCount;
	mySystem();
	mySystem(string filename);
	mySystem(string filename, int countN);
	mySystem(unsigned int count, int pd, int pu, int dd, int du, int md, int mu);
	void showInfo();
	vector<Body> getBody() {
		return allBodies;
	}
};

