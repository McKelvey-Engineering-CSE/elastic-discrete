#ifndef SHEDULER_H
#define SCHEDULER_H

/*************************************************************************

scheduler.h

This object contains the Scheduler object itself along with the scheduling
algorithm employed to schedule elastic tasks.


Class : shared_mem

		This class contains all of the scheduling algorithm logic
		as well as the actual driver code for deploying a derived schedule.

        This class contains a sub class "sched_pair" which is used in 
		a heap to monitor differnt task pairs. By default this inner
		class is disabled and prevented from being compiled. To enable it,
		define SCHED_PAIR_HEAP.

**************************************************************************/

#include <vector>
#include <algorithm>
#include "schedule.h"
#include <sched.h>
#include <signal.h>
#include <unistd.h>
#include "include.h"

class Scheduler{

	//structure for internal knapsack scheduler
	struct task_mode {
		double cpuLoss;
		double gpuLoss;
		size_t cores;
		size_t sms;
	};

	pid_t process_group;
	class Schedule schedule;
	size_t num_tasks;
	int num_CPUs;
	bool first_time;

	//each entry is a task with each item in the vector representing a mode
	static std::vector<std::vector<task_mode>> task_table;
	
public:

	//reserve the necessary space for the class (task) table
	Scheduler(int num_tasks_, int num_CPUs_) : process_group(getpgrp()), schedule("EFSschedule"), num_tasks(num_tasks_), num_CPUs(num_CPUs_), first_time(true) {

		//clear the vector of vectors (should retain static memory allocation)
		for (int i = 0; i < num_tasks_; i++)
			task_table.at(i).clear();
		task_table.clear();

 	}

	~Scheduler(){}

	void do_schedule(size_t maxCPU = 16, size_t maxSMS = 16);

	void setTermination();

	class Schedule * get_schedule();

	TaskData * add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_);
};


#endif
