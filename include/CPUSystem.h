#pragma once
#include <windows.h>
#include "mySystem.h"

class CPUSystem :
    public mySystem
{
public:
    CPUSystem(string filename) :mySystem(filename) {};
    CPUSystem(string filename, int countN) :mySystem(filename, countN) {};
    CPUSystem(unsigned int count, int pd, int pu, int dd, int du, int md, int mu) :mySystem(count, pd, pu, dd, du, md, mu) {};
    void iterate(int iterNum);
};

