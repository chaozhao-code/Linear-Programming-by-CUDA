#include <windows.h>
#include "PerformanceTimer.h"

#pragma comment(lib, "winmm.lib")


PerformanceTimer::PerformanceTimer(bool bCheckPerfHardware)
{
	// check whether we have performance timer	
	__int64 m_i64PerfFrequency;
	m_bPerfHardware = (bCheckPerfHardware ? QueryPerformanceFrequency((LARGE_INTEGER *) &m_i64PerfFrequency) != 0 : false);

	if (m_bPerfHardware)
	{
		QueryPerformanceCounter((LARGE_INTEGER *) &m_i64LastTime);		
		m_dblTimeScale  = 1.0 / m_i64PerfFrequency;
	}
	else
	{
		m_i64LastTime   = timeGetTime();
		m_dblTimeScale  = 0.001;
	}
}

PerformanceTimer::~PerformanceTimer()
{
}

void PerformanceTimer::StartTimer()
{
	if (m_bPerfHardware)
		QueryPerformanceCounter((LARGE_INTEGER *) &m_i64LastTime);
	else
		m_i64LastTime = timeGetTime();
}

double PerformanceTimer::GetTimeElapsed()
{
	__int64 i64CurrentTime;
	if (m_bPerfHardware)
		QueryPerformanceCounter((LARGE_INTEGER *) &i64CurrentTime);
	else
		i64CurrentTime = timeGetTime();
	return (i64CurrentTime - m_i64LastTime) * m_dblTimeScale;
}
