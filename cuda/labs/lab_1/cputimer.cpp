#include "cputimer.h"
#include <iostream>

CpuTimer::CpuTimer()
{

}

void CpuTimer::start()
{
    t1 = std::chrono::high_resolution_clock::now();
}

void CpuTimer::stop()
{
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - t1;
    std::cout <<"CPU    " << diff.count() * 1000 << " msecs\n";
}
