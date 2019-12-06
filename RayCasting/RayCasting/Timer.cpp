#include "Timer.h"

Timer::Timer()
{
	start = std::chrono::high_resolution_clock::now();
	stop = std::chrono::high_resolution_clock::now();
}

void Timer::Start()
{
	if (!isrunning)
	{
		start = std::chrono::high_resolution_clock::now();
		isrunning = true;
	}
}

void Timer::Stop()
{
	if (isrunning)
	{
		stop = std::chrono::high_resolution_clock::now();
		isrunning = false;
		AccumulateTime();
	}
}

void Timer::Restart()
{
	Reset();
	Start();
}

void Timer::Reset()
{
	accumulatedTime = 0;
	start = stop;
}

void Timer::AccumulateTime()
{
	if (!isrunning)
	{
		auto elapsed = std::chrono::duration<double, std::milli>(stop - start);
		accumulatedTime += elapsed.count();
	}
}

double Timer::GetMilisecondsElapsed()
{
	if (isrunning)
	{
		auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start);
		return accumulatedTime + elapsed.count();
	}
	else
	{
		auto elapsed = std::chrono::duration<double, std::milli>(stop - start);
		return accumulatedTime + elapsed.count();
	}
}