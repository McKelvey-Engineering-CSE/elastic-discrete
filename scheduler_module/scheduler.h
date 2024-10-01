#ifndef SHEDULER_H
#define SCHEDULER_H

/*************************************************************************

scheduler.h

This object contains the Scheduler object itself along with the scheduling
algorithm employed to schedule elastic tasks.


Class : shared_mem

		This class contains all of the scheduling algorithm logic.

		The current logic is a double knapsack problem that is solved
		dynamically. The knapsack problem is solved twice, once for
		CPUs and once for SMs. The solution is then used to determine
		the optimal mode for each task.

**************************************************************************/

#include <vector>
#include <algorithm>
#include "schedule.h"
#include <sched.h>
#include <signal.h>
#include <unistd.h>
#include "include.h"

#define DEFAULT_MAX_CPU 16
#define DEFAULT_MAX_SMS 16

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

	size_t maxSMS = DEFAULT_MAX_SMS;

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

	void do_schedule(size_t maxCPU = DEFAULT_MAX_CPU);

	void setTermination();

	class Schedule * get_schedule();

	TaskData * add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_);
};


#endif
