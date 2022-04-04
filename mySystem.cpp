#include "./include/mySystem.h"

mySystem::mySystem(unsigned int count, int pd, int pu, int dd, int du, int md, int mu) {
	srand((unsigned)time);
	this->bodyCount = count;
	while (count--) {
		float3 pos{ (float)((rand() % (pu - pd)) + pd), (float)((rand() % (pu - pd)) + pd), (float)((rand() % (pu - pd)) + pd) };
		float3 dir{ (float)((rand() % (du - dd)) + dd), (float)((rand() % (du - dd)) + dd), (float)((rand() % (du - dd)) + dd) };
		float mass = (float)((rand() % (mu - md)) + md);
		allBodies.emplace_back(Body(pos, dir, mass));
	}
}

mySystem::mySystem(string filename) {
	ifstream inFile(filename, ios::in);
	if (!inFile) {
		cout << "Can't open file:" << filename << endl;
		exit(-1);
	}
	string line, field;
	int count = 0;
	getline(inFile, line);
	while (getline(inFile, line)) {
		istringstream lineCon(line);
		getline(lineCon, field, ',');
		float x = atof(field.c_str());
		getline(lineCon, field, ',');
		float y = atof(field.c_str());
		getline(lineCon, field, ',');
		float z = atof(field.c_str());
		getline(lineCon, field, ',');
		float vx = atof(field.c_str());
		getline(lineCon, field, ',');
		float vy = atof(field.c_str());
		getline(lineCon, field, ',');
		float vz = atof(field.c_str());
		getline(lineCon, field, ',');
		float m = atof(field.c_str());
		getline(lineCon, field);
		int id = atof(field.c_str());
	
		float3 pos = { x, y, z };
		float3 dir = { vx, vy, vz };
		this->allBodies.emplace_back(Body(pos, dir, m));
		count++;
	}
	this->bodyCount = count;
	inFile.close();
}

mySystem::mySystem(string filename, int countN) {
	ifstream inFile(filename, ios::in);
	if (!inFile) {
		cout << "Can't open file:" << filename << endl;
		exit(-1);
	}
	string line, field;
	int count = 0;
	getline(inFile, line);
	while (getline(inFile, line) && countN--) {
		istringstream lineCon(line);
		getline(lineCon, field, ',');
		float x = atof(field.c_str());
		getline(lineCon, field, ',');
		float y = atof(field.c_str());
		getline(lineCon, field, ',');
		float z = atof(field.c_str());
		getline(lineCon, field, ',');
		float vx = atof(field.c_str());
		getline(lineCon, field, ',');
		float vy = atof(field.c_str());
		getline(lineCon, field, ',');
		float vz = atof(field.c_str());
		getline(lineCon, field, ',');
		float m = atof(field.c_str());
		getline(lineCon, field);
		int id = atof(field.c_str());

		float3 pos = { x, y, z };
		float3 dir = { vx, vy, vz };
		this->allBodies.emplace_back(Body(pos, dir, m));
		count++;
	}
	this->bodyCount = count;
	inFile.close();
}

void mySystem::showInfo() {
	cout << "========================================" << endl;

	cout << "count of bodies:" << this->bodyCount << endl;

	cout << "softerning factor:" << sf;
	cout << "\tgravitational constant:" << G;
	cout << "\ttime granularity:" << deltaT << endl;

	for (auto& itemI : allBodies) {
		itemI.showInfo();
	}
}