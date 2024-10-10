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

	//October 23 2023- moved struct to class along with comparison 
	//operators and locked behind flag due to seemingly no
	//functionality in code besides *possible* memory alignment?
	#ifdef SCHED_PAIR_HEAP

		class sched_pair {
			int index; //Index of the task
			double weight; //Potential benefit to the schedule

			sched_pair(int index_, double weight_) : index(index_), weight(weight_) {}

			//comparison operators
			bool operator>(const sched_pair& rhs);
			bool operator<(const sched_pair& rhs);
			bool operator==(const sched_pair& rhs);
			bool operator<=(const sched_pair& rhs);
			bool operator>=(const sched_pair& rhs);
			bool operator!=(const sched_pair& rhs);
		};

		std::vector<sched_pair> sched_heap;

	#endif

	pid_t process_group;
	class Schedule schedule;
	int num_tasks;
	int num_CPUs;
	bool first_time;

	//Dynamic Programming table. DP[d][l].first contains the optimal value
	//of trying to schedule the first l tasks on d CPUs.
	//DP[d][l].second is the modes of operation for each task.
	std::pair<double,std::vector<int>> DP[NUMCPUS+1][MAXTASKS+1];
	
public:

	//October 23 2023 - Removed the sched_pair heap due to seemingly no use. Memory is reserved for it though
	//so if it turns out that the memory reservation is being used for alignment, the solution for fixing the code is 
	//reenabling the SCHED_PAIR_HEAP flag
	Scheduler(int num_tasks_, int num_CPUs_) : process_group(getpgrp()), schedule("EFSschedule"), num_tasks(num_tasks_), num_CPUs(num_CPUs_), first_time(true) {
		#ifdef SCHED_PAIR_HEAP
			sched_heap.reserve(num_tasks);
		#endif

		for (int i = 0; i < NUMCPUS+1; i++) {
			for (int j = 0; j < MAXTASKS+1; j++) {
				DP[i][j].second.reserve(num_tasks);
			}
		}
 	}

	~Scheduler(){}

	void do_schedule();

	void setTermination();

	class Schedule * get_schedule();

	TaskData * add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_);
};


#endif
