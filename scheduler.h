#ifndef SHEDULER_H
#define SCHEDULER_H

#include <vector>
#include <algorithm>
#include "schedule.h"
#include <sched.h>
#include <signal.h>
#include <unistd.h>
#include "include.h"

struct sched_pair{
	int index; //Index of the task
	double weight; //Potential benefit to the schedule

	sched_pair(int index_, double weight_) : index(index_), weight(weight_) {}
};

bool operator>(const sched_pair& lhs, const sched_pair& rhs);
bool operator<(const sched_pair& lhs, const sched_pair& rhs);
bool operator==(const sched_pair& lhs, const sched_pair& rhs);
bool operator<=(const sched_pair& lhs, const sched_pair& rhs);
bool operator>=(const sched_pair& lhs, const sched_pair& rhs);
bool operator!=(const sched_pair& lhs, const sched_pair& rhs);

class Scheduler{
private:
	pid_t process_group;
	class Schedule schedule;
	int num_tasks;
	int num_CPUs;
	std::vector<sched_pair> sched_heap;
	bool first_time;

	//Dynamic Programming table. DP[d][l].first contains the optimal value
	//of trying to schedule the first l tasks on d CPUs.
	//DP[d][l].second is the modes of operation for each task.
	std::pair<double,std::vector<int>> DP[NUMCPUS+1][MAXTASKS+1];
	//std::vector<std::vector<std::pair<double,std::string>>> weights;
	
public:
	void do_schedule();
	Scheduler(int num_tasks_, int num_CPUs_) : process_group(getpgrp()), schedule("EFSschedule"), num_tasks(num_tasks_), num_CPUs(num_CPUs_), first_time(true) {
		sched_heap.reserve(num_tasks);
	}

	class Schedule * get_schedule();
	TaskData * add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_);

	~Scheduler(){
	}
};


#endif
