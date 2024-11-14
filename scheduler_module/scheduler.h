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
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include "include.h"

//NVIDIA headers
#ifdef __NVCC__
	
	#include "libsmctrl.h"

#endif

class Scheduler{

	enum resource_type {CORE_A, CORE_B};

	//structure for internal knapsack scheduler
	struct task_mode {
		double cpuLoss = 0.0;
		double gpuLoss = 0.0;
		int cores = 0;
		int sms = 0;
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
	struct Edge {
		int to_node;
		int x_amount;  // amount of resource x transferred (0 if none)
		int y_amount;  // amount of resource y transferred (0 if none)
	};

	struct Node {
		int id;
		int x, y;  // resources (negative means needed)
		std::vector<Edge> edges;  // outgoing edges with resource amounts
	};

	pid_t process_group;
	class Schedule schedule;
	size_t num_tasks;
	int num_CPUs;
	bool first_time;

	uint32_t GPC_size;
	uint64_t* TPC_to_GPC_masks;

	size_t bound_GPU_device = 0;

	bool barrier = true;

	bool FPTAS = true;

	//each entry is a task with each item in the vector representing a mode
	static std::vector<std::vector<task_mode>> task_table;

	//each entry is a task mode that the corresponding task was last running in
	std::vector<Scheduler::task_mode> previous_modes;

	//each entry corresponds to a task that dictates how it will be processed in the knapsack algorithm
	static std::vector<int> class_mappings;

	//each entry corresponds to a core which is held by a task
	std::vector<int> free_cores_A;
	std::vector<int> free_cores_B;
	
public:

	//reserve the necessary space for the class (task) table
	Scheduler(int num_tasks_, int num_CPUs_, bool explicit_sync, bool FPTAS_) : process_group(getpgrp()), schedule("EFSschedule"), num_tasks(num_tasks_), num_CPUs(num_CPUs_), first_time(true), barrier(explicit_sync), FPTAS(FPTAS_) {

		previous_modes.reserve(100);

		//clear the vector of vectors (should retain static memory allocation)
		for (int i = 0; i < num_tasks_; i++)
			task_table.at(i).clear();
		task_table.clear();

		free_cores_A.reserve(num_CPUs);
		free_cores_B.reserve(NUMGPUS);

		//reserve the backtrack table

 	}

	~Scheduler(){}

	void do_schedule(size_t maxCPU = NUMCPUS - 1);

	std::vector<int> sort_classes(std::vector<int> items_in_candidate);

	void setTermination();

	class Schedule * get_schedule();

	void set_FPTAS();

	int get_num_tasks();

	bool has_cycle(const std::unordered_map<int, Node>& nodes, int start);

	bool build_resource_graph(std::vector<std::pair<int, int>> resource_pairs, 
                        std::unordered_map<int, Node>& nodes, std::unordered_map<int, Node>& static_nodes);

	void print_graph(const std::unordered_map<int, Node>& nodes, std::unordered_map<int, Node> static_nodes);

	TaskData * add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_);
};


#endif
