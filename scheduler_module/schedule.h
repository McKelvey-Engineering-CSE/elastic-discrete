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

#include "shared_mem.h"
#include "taskData.h"
#include "include.h"

#include <atomic>

const int INVALID_INDEX=-1;

class Schedule : public shared_mem {

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
