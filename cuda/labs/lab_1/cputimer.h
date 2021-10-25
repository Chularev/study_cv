#ifndef CPUTIMER_H
#define CPUTIMER_H

#include <chrono>

class CpuTimer
{
public:
    CpuTimer();
    void start();
    void stop();
private:
     std::chrono::time_point<std::chrono::system_clock> t1;
};

#endif // CPUTIMER_H
