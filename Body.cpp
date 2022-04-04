#include "./include/Body.h"
using namespace std;

Body::Body(float3 pos, float3 dir, float mass)
{
	this->pos = pos;
	this->dir = dir;
	this->mass = mass;
	this->acc = float3{ 0.0f,0.0f,0.0f };
}

Body::~Body()
{
}

void Body::showInfo()
{
	cout << "Mass:" << mass;
	cout << "\tPosition:" << pos.x << "," << pos.y << "," << pos.z;
	cout << "\tDirection:" << dir.x << "," << dir.y << "," << dir.z << endl;
}

void Body::updateDir(float3 newDir)
{
	this->dir = newDir;
}

void Body::updateDir(float newX, float newY, float newZ)
{
	this->dir.x = newX;
	this->dir.y = newY;
	this->dir.z = newZ;
}

void Body::updatePos(float3 newPos)
{
	this->pos = newPos;
}

void Body::updatePos(float newX, float newY, float newZ)
{
	this->pos.x = newX;
	this->pos.y = newY;
	this->pos.z = newZ;
}

void Body::updateAcc(float3 newAcc)
{
	this->acc = newAcc;
}

void Body::updateAcc(float newX, float newY, float newZ)
{
	this->acc.x = newX;
	this->acc.y = newY;
	this->acc.z = newZ;
}
