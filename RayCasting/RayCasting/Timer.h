#pragma once
#include <chrono>

class Timer
{
public:
	Timer();
	void Start();
	void Stop();
	void Reset();
	void Restart();
	double GetMilisecondsElapsed();

private:
	void AccumulateTime();
	double accumulatedTime = 0.0;
	bool isrunning = false;

#ifdef _WIN32
	std::chrono::time_point<std::chrono::steady_clock> start;
	std::chrono::time_point<std::chrono::steady_clock> stop;
#else
	std::chrono::time_point<std::chrono::system_clock> start;
	std::chrono::time_point<std::chrono::system_clock> stop;
#endif
};
