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
#include <sched.h>
#include <signal.h>
#include <unistd.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cerrno>
#include <float.h>
#include <map>
#include <tuple>
#include <cstring>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <stack>

#include "taskData.h"
#include "print_module.h"
#include "include.h"
#include "schedule.h"

//NVIDIA headers
#ifdef __NVCC__
	
	#include <cuda.h>
	#include <cuda_runtime.h>

#endif

class Scheduler{

	enum resource_type {CORE_A, CORE_B, CORE_C, CORE_D};

	//structure for internal knapsack scheduler
	struct task_mode {
		double cpuLoss = 0.0;
		double gpuLoss = 0.0;
		double cpuCLoss = 0.0;
		double gpuDLoss = 0.0;
		int processors_A = 0;
		int processors_B = 0;
		int processors_C = 0;
		int processors_D = 0;
		bool unsafe_mode = false;
	};

	//structure for item map cause I'm lazy
	struct item_map {

		//0 for A core
		//1 for B core
		//2 for C core
		//3 for D core
		int resource_type = 0;

		int resource_amount = -1;
		
		int task_id = -1;
	};

	//structure for RAG vertices
	struct Edge {
		int to_node;
		int a_amount;  // amount of resource A transferred (0 if none)
		int b_amount;  // amount of resource B transferred (0 if none)
		int c_amount;  // amount of resource C transferred (0 if none)
		int d_amount;  // amount of resource D transferred (0 if none)
	};

	struct Node {
		int id;
		int a, b, c, d;  // resources (negative means needed)
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

	//each entry corresponds to a processor which is held by a task
	std::vector<int> free_processors_A;
	std::vector<int> free_processors_B;
	std::vector<int> free_processors_C;
	std::vector<int> free_processors_D;

	//taskData structure for the free resource pool
	TaskData free_pool;

	//table for storing the combination for unsafe tasks later
	int* unsafe_table;

	//max loss for a given task system
	double max_loss = 0.0;

	//pointers for cuda memory
	double* d_losses;
	double* d_final_loss;
	double* cautious_d_final_loss;
	int* d_final_solution;
	int* d_current_task_modes;
	int* cautious_d_final_solution;
	int* d_uncooperative_tasks;

	#ifdef __NVCC__
		
		CUgreenCtx green_ctx;

		CUcontext primary_scheduler_context;

		cudaStream_t scheduler_stream;

		cudaStream_t cautious_stream;

		void create_scheduler_stream();

	#endif

	//FIXME: REPLACE WITH REAL LOCK
	bool scheduler_running = false;
	
public:

	//reserve the necessary space for the class (task) table
	Scheduler(int num_tasks_, int num_CPUs_, bool explicit_sync, bool FPTAS_) : process_group(getpgrp()), schedule("EFS"), num_tasks(num_tasks_), num_CPUs(num_CPUs_), first_time(true), barrier(explicit_sync), FPTAS(FPTAS_) {

		previous_modes.reserve(100);

		//clear the vector of vectors (should retain static memory allocation)
		for (int i = 0; i < num_tasks_; i++)
			task_table.at(i).clear();
		task_table.clear();

		free_processors_A.reserve(num_CPUs);
		free_processors_B.reserve(NUM_PROCESSOR_B);
		free_processors_C.reserve(NUM_PROCESSOR_C);
		free_processors_D.reserve(NUM_PROCESSOR_D);

		//reserve the backtrack table

 	}

	~Scheduler(){}

	void generate_unsafe_combinations(size_t maxCPU = NUM_PROCESSOR_A - 1);

	void do_schedule(size_t maxCPU, bool check_max_possible = false);

	std::vector<int> sort_classes(std::vector<int> items_in_candidate);

	void setTermination();

	class Schedule * get_schedule();

	void set_FPTAS();

	int get_num_tasks();

	bool check_if_scheduler_running();

	bool has_cycle(const std::unordered_map<int, Node>& nodes, int start);

	bool build_resource_graph(std::vector<std::tuple<int, int, int, int>> resource_pairs, 
                        std::unordered_map<int, Node>& nodes, std::unordered_map<int, Node>& static_nodes, std::vector<int>& task_modes,
						std::vector<int> lowest_modes);

	void execute_resource_allocation_graph(std::vector<std::tuple<int, int, int, int>> resource_pairs, 
                        std::unordered_map<int, Node>& nodes);

	void print_graph(const std::unordered_map<int, Node>& nodes, std::unordered_map<int, Node> static_nodes);

	TaskData * add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_, timespec * cpu_C_work_, timespec * cpu_C_span_, timespec * cpu_C_period_, timespec * gpu_D_work_, timespec * gpu_D_span_, timespec * gpu_D_period_, bool safe);
};


#endif
