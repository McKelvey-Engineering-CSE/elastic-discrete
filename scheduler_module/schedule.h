#ifndef _SHARED_SCHEDULE_H
#define  _SHARED_SCHEDULE_H

/*************************************************************************

schedule.h

TODO: WRITE BETTER DESCRIPTION

This object inherits from and creates the shared memory segment for the 
scheduling algorithm. It also manages the interaction between the scheduler, 
tasks, and memory segments.


Class : Schedule

		This class inherits from the Shared memory class and implements
		the maintenance of the shared memory mappings and the interaction
		between the scheduler algorithm and the tasks themselves

**************************************************************************/

#include "memory_allocator.h"
#include "taskData.h"
#include "include.h"

#include <string.h>
#include <atomic>

const int INVALID_INDEX=-1;

struct schedule_object {
	
	int next_task = 0;
	TaskData task[MAXTASKS];

	schedule_object() {}

};

class Schedule {

	schedule_object* underlying_object = nullptr;

	std::string name;

	bool owner = false;

public: 
	Schedule(std::string name_, bool create = true);	
	~Schedule();

	bool needs_change();
	void set_needs_change(bool change);
	TaskData * add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_, timespec * cpu_C_work_, timespec * cpu_C_span_, timespec * cpu_C_period_, timespec * gpu_D_work_, timespec * gpu_D_span_, timespec * gpu_D_period_, bool safe);
	int count();
	TaskData * get_task(int n);

	void setTermination();
	
};
#endif
