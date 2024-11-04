#ifndef _TASKDATA_H
#define  _TASKDATA_H

/*************************************************************************

taskData.h

This object returns a bunch of data about a given task... need to look at code
more to come up with a better description than that


Class : TaskData

		Class that stores information related to a given task, as well as providing
		getters and setters to modify those data points/values
        
**************************************************************************/

#include <algorithm>
#include <assert.h>
#include <stdlib.h> 
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <array>
#include "timespec_functions.h"
#include "include.h"
#include "print_module.h"

class TaskData{

private:

	bool is_pure_cpu_task = true;

	static int counter;
	int index; //unique identifier
	
	bool changeable;
	bool can_reschedule;
	int num_adaptations;

	//These are read in. James 9/7/18
	double elasticity;	
	int num_modes;
	timespec work[MAXMODES];
	timespec span[MAXMODES];
	timespec period[MAXMODES];
	int CPUs[MAXMODES];

	timespec GPU_work[MAXMODES];
	timespec GPU_span[MAXMODES];
	timespec GPU_period[MAXMODES];
	int GPUs[MAXMODES];

	//These are computed.
	double max_utilization;
	int max_CPUs;
	int min_CPUs;

	int max_GPUs;
	int min_GPUs;

	int CPUs_gained;
	int GPUs_gained;

	int practical_max_CPUs;	
	int current_lowest_CPU;

	int practical_max_GPUs;	
	int current_lowest_GPU = 0;

	double percentage_workload;

	timespec current_period;
	timespec current_work;
	timespec current_span;
	double current_utilization;

	int current_CPUs;
	int previous_CPUs;

	int current_GPUs;
	int previous_GPUs;

	int permanent_CPU;

	int current_mode;
	timespec max_work;

	//TPC mask
	__uint128_t TPC_mask = 0;

	//CPU core mask
	__uint128_t CPU_mask = 0;

	//updated variables
	bool mode_transitioned = false;

	//these variables are set by the scheduler to denote
	//how many of our resources we are supposed to return
	int cpus_to_return = 0;
	int gpus_to_return = 0;

	//these denote the number of tasks we are looking for
	//when we are collecting our resources. The scheduler 
	//will use these more than we will as tasks
	int other_tasks_giving_cpus = 0;
	int other_tasks_giving_gpus = 0;

	//these are the indicies of processors we are gaining
	//and which processors they came from

	//the position MAXTASKS + 1 is the free resource pool
	//it is treated as a task for no reason other than 
	//consistency
	int cpus_granted_from_other_tasks [MAXTASKS + 1][NUMCPUS + 1];
	int gpus_granted_from_other_tasks [MAXTASKS + 1][NUMGPUS + 1];

public:

	TaskData(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_);

	~TaskData();

	int get_index();
	double get_elasticity();
	double get_percentage_workload();
	bool get_changeable();
	
	double get_max_utilization();
	int get_max_CPUs();
	int get_min_CPUs();

	int get_practical_max_CPUs();
	void set_practical_max_CPUs(int new_value);

	timespec get_current_period();
	timespec get_current_work();
	timespec get_current_span();

	int get_current_CPUs();
	int get_current_lowest_CPU();
	
	int get_CPUs_gained();
	void set_CPUs_gained(int new_CPUs_gained);

	void set_current_mode(int new_mode, bool disable);
	int get_current_mode();

	void reset_changeable();
	void set_current_lowest_CPU(int _lowest);

	int get_permanent_CPU();
	void set_permanent_CPU(int perm);
	
	int get_num_adaptations();
	void set_num_adaptations(int new_num);

	timespec get_work(int index);
	timespec get_span(int index);
	timespec get_period(int index);
	int get_CPUs(int index);

	//GPU functions that can be compiled regardless of compiler
	timespec get_GPU_work(int index);
	timespec get_GPU_span(int index);
	timespec get_GPU_period(int index);

	int get_GPUs(int index);
	int get_max_GPUs();
	int get_min_GPUs();
	int get_current_GPUs();

	void set_practical_max_GPUs(int new_value);
	int get_practical_max_GPUs();
	void set_current_lowest_GPU(int _lowest);
	int get_current_lowest_GPU();

	int get_GPUs_gained();
	void set_GPUs_gained(int new_GPUs_gained);

	bool pure_cpu_task();

	//reworking all the CPU and GPU handoff functions
	//NOTE: all return functions will work from the 
	//highest CPU/SM unit we have down until we run
	//out of CPUs/SMs to return
	void set_CPUs_change(int num_cpus_to_return);

	void set_GPUs_change(int num_gpus_to_return);

	int get_CPUs_change();

	int get_GPUs_change();

	//function to check if this task has transitioned
	//to a new mode yet
	bool check_mode_transition();

	void set_mode_transition(bool state);

	//functions to work with static vector of CPU indices
	int pop_back_cpu();

	int push_back_cpu(int index);

	int pop_back_gpu();

	int push_back_gpu(int index);

	int get_cpu_at_index(int index);

	std::vector<int> get_cpu_owned_by_process();

	std::vector<int> get_gpu_owned_by_process();

	//retrieve the number of CPUs or GPUs we have been given	
	std::vector<std::pair<int, std::vector<int>>> get_cpus_granted_from_other_tasks();

	std::vector<std::pair<int, std::vector<int>>> get_gpus_granted_from_other_tasks();

	//give CPUs or GPUs to another task
	void set_cpus_granted_from_other_tasks(std::pair<int, std::vector<int>> entry);

	void set_gpus_granted_from_other_tasks(std::pair<int, std::vector<int>> entry);

	//make a function which clears these vectors like they are cleared in the constructor
	void clear_cpus_granted_from_other_tasks();

	void clear_gpus_granted_from_other_tasks();

	__uint128_t get_cpu_mask();

	__uint128_t get_gpu_mask();

};

#endif
