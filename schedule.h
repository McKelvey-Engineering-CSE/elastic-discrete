#ifndef _SHARED_SCHEDULE_H
#define  _SHARED_SCHEDULE_H

#include "sharedMem.h"
#include "taskData.h"
#include "include.h"

#include <atomic>

const int INVALID_INDEX=-1;

class Schedule : sharedMem {

public: 
	Schedule(std::string name_);	
	~Schedule();

	bool needs_change();
	void set_needs_change(bool change);
	TaskData * add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_);
	int count();
	TaskData * get_task(int n);
	
};
#endif
