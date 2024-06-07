// PerformanceTimer.h - High Resolution Timer
// Since not every PC has performance hardware, we have fallback mechanism
// to timeGetTime()
//

#ifndef __PERFORMANCETIMER_H__
#define __PERFORMANCETIMER_H__

class PerformanceTimer
{		
	__int64 m_i64LastTime;
	double	m_dblTimeScale;
	bool	m_bPerfHardware;

public:
	PerformanceTimer(bool bCheckPerfHardware = true);
	~PerformanceTimer();

	void	StartTimer();
	double	GetTimeElapsed();
	bool	IsUsingPerfHardware() const { return m_bPerfHardware; }
};

#endif