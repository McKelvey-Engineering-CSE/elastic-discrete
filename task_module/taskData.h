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
#include "timespec_functions.h"
#include "include.h"
#include "print_module.h"

//NVIDIA headers
#ifdef __NVCC__
	
	#include <cuda.h>
	#include <cuda_runtime.h>

	#define NVRTC_SAFE_CALL(x)                                        \
	do {                                                              \
	nvrtcResult result = x;                                           \
	if (result != NVRTC_SUCCESS) {                                    \
		std::cerr << "\nerror: " #x " failed with error "             \
					<< nvrtcGetErrorString(result) << '\n';           \
		exit(1);                                                      \
	}                                                                 \
	} while(0)

	#define CUDA_SAFE_CALL(x)                                         \
	do {                                                              \
	CUresult result = x;                                              \
	if (result != CUDA_SUCCESS) {                                     \
		const char *msg;                                              \
		cuGetErrorName(result, &msg);                                 \
		std::cerr << "\nerror: " #x " failed with error "             \
					<< msg << '\n';                                   \
		exit(1);                                                      \
	}                                                                 \
	} while(0)

	#define CUDA_NEW_SAFE_CALL(x)                                     \
	do {                                                              \
	cudaError_t result = x;                                           \
	if (result != cudaSuccess) {                                      \
		std::cerr << "\nerror: " #x " failed with error "             \
					<< cudaGetErrorName(result) << '\n';              \
		exit(1);                                                      \
	}                                                                 \
	} while(0)

	#define NVJITLINK_SAFE_CALL(h,x)                                  \
	do {                                                              \
	nvJitLinkResult result = x;                                       \
	if (result != NVJITLINK_SUCCESS) {                                \
		std::cerr << "\nerror: " #x " failed with error "             \
					<< result << '\n';                                \
		size_t lsize;                                                 \
		result = nvJitLinkGetErrorLogSize(h, &lsize);                 \
		if (result == NVJITLINK_SUCCESS && lsize > 0) {               \
			char *log = (char*)malloc(lsize);                         \
			result = nvJitLinkGetErrorLog(h, log);                    \
			if (result == NVJITLINK_SUCCESS) {                        \
				std::cerr << "error: " << log << '\n';                \
				free(log);                                            \
			}                                                         \
		}                                                             \
		exit(1);                                                      \
	}                                                                 \
	} while(0)

#endif

class TaskData{
private:

	//updated in constructor, left with 16 for the event this 
	//is compiled with g++ and someone forgets to actually update the task.yaml file
	int NUMGPUS = 16;

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

	double practical_max_utilization;

	int practical_max_CPUs;	
	int current_lowest_CPU;

	int practical_max_GPUs;	
	int current_lowest_GPU;

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

	int active_cpus[NUMCPUS + 1];
	int passive_cpus[NUMCPUS + 1];

	int give_CPU[MAXTASKS];
	bool transfer_CPU[MAXTASKS][NUMCPUS + 1];
	bool receive_CPU[MAXTASKS][NUMCPUS + 1];

	//all are the size of NUMGPU + 1
	int* active_gpus;
	int* passive_gpus;

	int give_GPU[MAXTASKS];
	bool* transfer_GPU[MAXTASKS];
	bool* receive_GPU[MAXTASKS];

	//GPU SM management variables
	#ifdef __NVCC__
		
		//assume we have the largest GPU (in theory) possible
		CUdevResource total_TPCs[144];

		//assume we have the largest GPU (in theory) possible
		CUdevResource our_TPCs[144];

		//and store the actual number of TPCs we have
		unsigned int num_TPCs = 1000;

		//store how many TPCs we are granted
		int granted_TPCs = 0;

		//TPC mask
		__uint128_t TPC_mask;

	#endif

public:

	TaskData(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_);

	~TaskData();

	int get_num_modes();	
	int get_index();
	double get_elasticity();
	double get_percentage_workload();
	bool get_changeable();
	
	double get_max_utilization();
	int get_max_CPUs();
	int get_min_CPUs();

	timespec get_max_work();

	double get_practical_max_utilization();
	int get_practical_max_CPUs();
	void set_practical_max_CPUs(int new_value);

	timespec get_current_period();
	timespec get_current_work();
	timespec get_current_span();

    double get_current_utilization();
	int get_current_CPUs();
	int get_current_lowest_CPU();
	
	int get_CPUs_gained();
	void set_CPUs_gained(int new_CPUs_gained);

	int get_previous_CPUs();
	void set_previous_CPUs(int new_prev);

	void set_current_mode(int new_mode, bool disable);
	int get_current_mode();

	void reset_changeable();
	void set_current_lowest_CPU(int _lowest);

	void update_give(int index, int value);
	int gives(int index);

	bool transfers(int task, int CPU);
	void set_transfer(int task, int CPU, bool value);

	bool receives(int task, int CPU);
	void set_receive(int task, int CPU, bool value);

	int get_permanent_CPU();
	void set_permanent_CPU(int perm);
	
	void set_active_cpu(int i);
	void clr_active_cpu(int i);
	void set_passive_cpu(int i);
	void clr_passive_cpu(int i);
	bool get_active_cpu(int i);
	bool get_passive_cpu(int i);

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

	int get_total_TPC_count();

	int get_GPUs_gained();
	void set_GPUs_gained(int new_CPUs_gained);


	//related GPU functions
	#ifdef __NVCC__
		
		void set_TPC_mask(__uint128_t TPCs_to_enable);

		__uint128_t get_TPC_mask();

		CUcontext create_partitioned_context(std::vector<int> tpcs_to_add);

		__uint128_t make_TPC_mask(std::vector<int> TPCs_to_add);

	#endif

};

#endif
