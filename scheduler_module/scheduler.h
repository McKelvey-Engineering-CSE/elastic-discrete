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

//NVIDIA headers
#ifdef __NVCC__
	
	#include "libsmctrl.h"

#endif

#define DEFAULT_MAX_CPU 16
#define DEFAULT_MAX_SMS 16

class Scheduler{

	enum resource_type {CORE_A, CORE_B};

	//structure for internal knapsack scheduler
	struct task_mode {
		double cpuLoss;
		double gpuLoss;
		size_t cores;
		size_t sms;
	};

	//structure for item map cause I'm lazy
	struct item_map {

		//0 for A core
		//1 for B core
		int resource_type = 0;

		int resource_amount = -1;
		
		int task_id = -1;
	};

	//structure for RAG vertices
	struct vertex {

		int core_A = 0;
		int core_B = 0;

		int task_id = -1;

		std::vector<item_map> children;
	};

	pid_t process_group;
	class Schedule schedule;
	size_t num_tasks;
	int num_CPUs;
	bool first_time;

	uint32_t GPC_size;
	uint64_t* TPC_to_GPC_masks;

	size_t bound_GPU_device = 0;

	size_t maxSMS = DEFAULT_MAX_SMS;

	bool barrier = true;

	//each entry is a task with each item in the vector representing a mode
	static std::vector<std::vector<task_mode>> task_table;

	//each entry is a task mode that the corresponding task was last running in
	static std::vector<Scheduler::task_mode> previous_modes;

	//each entry corresponds to a task that dictates how it will be processed in the knapsack algorithm
	static std::vector<int> class_mappings;
	
public:

	//reserve the necessary space for the class (task) table
	Scheduler(int num_tasks_, int num_CPUs_) : process_group(getpgrp()), schedule("EFSschedule"), num_tasks(num_tasks_), num_CPUs(num_CPUs_), first_time(true) {

		//clear the vector of vectors (should retain static memory allocation)
		for (int i = 0; i < num_tasks_; i++)
			task_table.at(i).clear();
		task_table.clear();

		//Fetch GPC information if we are compiling ith nvcc
		#ifdef __NVCC__

			libsmctrl_get_gpc_info(&GPC_size, &TPC_to_GPC_masks, bound_GPU_device);

		#endif
 	}

	~Scheduler(){}

	void do_schedule(size_t maxCPU = DEFAULT_MAX_CPU);

	std::vector<int> sort_classes(std::vector<int> items_in_candidate);

	void build_RAG(std::vector<int> current_solution, std::vector<std::vector<vertex>>& final_RAG);

	bool check_for_cycles(std::vector<int> current_solution);

	void setTermination();

	class Schedule * get_schedule();

	TaskData * add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_);
};


#endif
