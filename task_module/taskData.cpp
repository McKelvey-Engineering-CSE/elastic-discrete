#include "taskData.h"	

TaskData::TaskData(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, 
														timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_) : 	
																													
																													index(counter++), changeable(true), 
																													can_reschedule(false), num_adaptations(0),  
																													elasticity(elasticity_), num_modes(num_modes_), 
																													max_utilization(0), max_CPUs(0), min_CPUs(NUMCPUS), 
																													max_GPUs(0), min_GPUs(NUMCPUS),  
																													CPUs_gained(0), practical_max_utilization(max_utilization),  
																													practical_max_CPUs(max_CPUs), current_lowest_CPU(-1), 
																													percentage_workload(1.0), current_period({0,0}), 
																													current_work({0,0}), current_span({0,0}), 
																													current_utilization(0.0), current_CPUs(0), previous_CPUs(0), 
																													permanent_CPU(-1), current_mode(0), max_work({0,0}){
	
	if (num_modes > MAXMODES){

		print_module::print(std::cerr, "ERROR: No task can have more than ", MAXMODES,  " modes.\n");
		kill(0, SIGTERM);
	
	}

	//read in all the task parameters
	for (int i = 0; i < num_modes; i++){

		//CPU parameters
		work[i] = *(work_ + i); 
		span[i] = *(span_ + i); 
		period[i] = *(period_ + i); 

		//GPU parameters
		GPU_work[i] = *(gpu_work_ + i);
		GPU_span[i] = *(gpu_span_ + i);
		GPU_period[i] = *(gpu_period_ + i);
	
	}			

	for (int i = 0; i < num_modes; i++)
		print_module::print(std::cout, work[i], " ", span[i], " ", period[i], "\n");	

	timespec numerator;
	timespec denominator;

	//determine resources
	for (int i = 0; i < num_modes; i++){

		//CPU resources
		if (work[i] / period[i] > max_utilization)
			max_utilization = work[i] / period[i];

		ts_diff(work[i], span[i], numerator);
		ts_diff(period[i], span[i], denominator);

		CPUs[i] = (int)ceil(numerator / denominator);
	
		if (CPUs[i] > max_CPUs)
			max_CPUs = CPUs[i];

		if (CPUs[i] < min_CPUs)
			min_CPUs = CPUs[i];

		if (work[i] > max_work)
			max_work = work[i];

		//GPU resources
		if (GPU_work[i] / GPU_period[i] > max_utilization)
			max_utilization = GPU_work[i] / GPU_period[i];

		ts_diff(GPU_work[i], GPU_span[i], numerator);
		ts_diff(GPU_period[i], GPU_span[i], denominator);

		GPUs[i] = (int)ceil(numerator / denominator);

		if (GPUs[i] > max_GPUs)
			max_GPUs = GPUs[i];

		if (GPUs[i] < min_GPUs)
			min_GPUs = GPUs[i];

	}

	current_CPUs = min_CPUs;
	current_GPUs = min_GPUs;

	for (int i = 0; i < MAXTASKS; i++){
		
		give[i] = 0;

		for (int j = 1; j <= NUMCPUS; j++){

			transfer[i][j] = false;
			receive[i][j] = false;
			active_cpus[i] = false;
			passive_cpus[i] = false;

		}
	}

	#ifdef __NVCC__

		CUdevResource initial_resources;

		//init device driver
		CUDA_SAFE_CALL(cuInit(0));

		//fill the initial descriptor
		CUDA_SAFE_CALL(cuDeviceGetDevResource(0, &initial_resources, CU_DEV_RESOURCE_TYPE_SM));

		//get the individual TPC slices
		CUDA_SAFE_CALL(cuDevSmResourceSplitByCount(total_TPCs, &num_TPCs, &initial_resources, NULL, CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));

		std::cout << "Maximum TPCs available: " << num_TPCs << " TPCs.\n";

	#endif

}

TaskData::~TaskData(){}

int TaskData::counter = 0;	        

int TaskData::get_index(){

	return index;

}

int TaskData::get_CPUs_gained(){

	return CPUs_gained;

}

void TaskData::set_CPUs_gained(int new_CPUs_gained){

	CPUs_gained = new_CPUs_gained;

}

int TaskData::get_previous_CPUs(){

	return previous_CPUs;

}

void TaskData::set_previous_CPUs(int new_prev){

	previous_CPUs = new_prev;

}

void TaskData::update_give(int index, int value){

	give[index]=value;

}

int TaskData::gives(int index){

	return give[index];

}

timespec TaskData::get_max_work(){

	return max_work;

}

double TaskData::get_elasticity(){

	return elasticity;

}

double TaskData::get_max_utilization(){

	return max_utilization;

}

int TaskData::get_max_CPUs(){

	return max_CPUs;

}

int TaskData::get_min_CPUs(){

	return min_CPUs;

}

//add the needed getters and setters for the gpu parameters
timespec TaskData::get_GPU_work(int index){

	return GPU_work[index];

}

timespec TaskData::get_GPU_span(int index){

	return GPU_span[index];

}

timespec TaskData::get_GPU_period(int index){

	return GPU_period[index];

}

int TaskData::get_GPUs(int index){

	return GPUs[index];

}

int TaskData::get_current_GPUs(){

	return current_GPUs;

}

int TaskData::get_max_GPUs(){

	return max_GPUs;

}

int TaskData::get_min_GPUs(){

	return min_GPUs;

}

timespec TaskData::get_current_span(){

	return current_span;

}

timespec TaskData::get_current_work(){

	return current_work;

}	

timespec TaskData::get_current_period(){

	return current_period;

}

double TaskData::get_current_utilization(){

	return current_utilization;

}

int TaskData::get_current_CPUs(){

	return current_CPUs;

}

double TaskData::get_percentage_workload(){

	return percentage_workload;

}

bool TaskData::get_changeable(){

	return changeable;

}

int TaskData::get_current_lowest_CPU(){

	return current_lowest_CPU;

}

double TaskData::get_practical_max_utilization(){

	return practical_max_utilization;

}

void TaskData::set_practical_max_CPUs(int new_value){

	practical_max_CPUs = new_value;

}

int TaskData::get_practical_max_CPUs(){

	return practical_max_CPUs;

}

void TaskData::set_current_lowest_CPU(int _lowest){

	current_lowest_CPU = _lowest;

}

void TaskData::set_practical_max_GPUs(int new_value){

	practical_max_GPUs = new_value;

}

int TaskData::get_practical_max_GPUs(){

	return practical_max_GPUs;

}

void TaskData::set_current_lowest_GPU(int _lowest){

	current_lowest_GPU = _lowest;

}

int TaskData::get_current_lowest_GPU(){

	return current_lowest_GPU;

}

void TaskData::set_active_cpu(int i){

	active_cpus[i] = true;

}

void TaskData::clr_active_cpu(int i){

	active_cpus[i] = false;

}

void TaskData::set_passive_cpu(int i){

	passive_cpus[i] = true;

}

void TaskData::clr_passive_cpu(int i){

	passive_cpus[i] = false;

}

bool TaskData::get_active_cpu(int i){

	return active_cpus[i];

}

bool TaskData::get_passive_cpu(int i){
	
	return passive_cpus[i];

}

void TaskData::set_current_mode(int new_mode, bool disable)
{
	if (new_mode >= 0 && new_mode < num_modes){

		current_mode = new_mode;
		current_work = work[current_mode];
		current_span = span[current_mode];
		current_period = period[current_mode];
		current_utilization = current_work/current_period;
		percentage_workload = current_work/max_work;
		previous_CPUs = current_CPUs;
		current_CPUs = CPUs[current_mode];
		changeable = (disable) ? false : true;
		
	}

	else{

		print_module::print(std::cerr, "Error: Task ", get_index(), " was told to go to invalid mode ", new_mode, ". Ignoring.\n");
	
	}
}

int TaskData::get_current_mode(){

	return current_mode;

}

void TaskData::reset_changeable(){

	changeable = true;

}

bool TaskData::transfers(int task, int CPU){

	return transfer[task][CPU];

}
	
void TaskData::set_transfer(int task, int CPU, bool value){

	transfer[task][CPU] = value;

}

bool TaskData::receives(int task, int CPU){

	return receive[task][CPU];

}

void TaskData::set_receive(int task, int CPU, bool value){

	receive[task][CPU] = value;

}

int TaskData::get_permanent_CPU(){

	return permanent_CPU;

}
	
void TaskData::set_permanent_CPU(int perm){

	permanent_CPU = perm;

}

int TaskData::get_num_adaptations(){

	return num_adaptations;

}

void TaskData::set_num_adaptations(int new_num){

	num_adaptations = new_num;

}

int TaskData::get_num_modes(){

	return num_modes;

}

timespec TaskData::get_work(int index){

	return work[index];

}
timespec TaskData::get_span(int index){

	return span[index];

}
timespec TaskData::get_period(int index){

	return period[index];

}
int TaskData::get_CPUs(int index){

	return CPUs[index];

}

//related GPU functions
#ifdef __NVCC__

	//this function will be used to create a mask of TPCs that are available to the task
	__uint128_t TaskData::make_TPC_mask(std::vector<int> TPCs_to_add){

		//create a mask
		__uint128_t mask = 0;

		//loop over each TPC in the vector and set the corresponding bit in the mask
		for (unsigned int i = 0; i < TPCs_to_add.size(); i++)
			mask |= (1 << TPCs_to_add[i]);

		//return the mask
		return mask;

	}
	
	//sets a created mask to the mask for the given task
	void TaskData::set_TPC_mask(__uint128_t TPCs_to_enable){

		//keep track of which sms we have been granted
		TPC_mask = TPCs_to_enable;

		//loop over each bit in TPCs_to_enable and print each bit that is set equal to 1
		granted_TPCs = 0;
		for (int i = 0; i < 128; i++)
			if (TPCs_to_enable & (1 << i))
				granted_TPCs += 1;

		//grab TPCs needed
		for (int i = 0; i < 128; i++)
			if (TPCs_to_enable & (1 << i))
				our_TPCs[i] = total_TPCs[i];

	}

	//returns the mask for the given task
	__uint128_t TaskData::get_TPC_mask(){

		return TPC_mask;

	}

	//quick function to create a SM context for the task
	CUcontext TaskData::create_partitioned_context(std::vector<int> tpcs_to_add){

		//output mask
		CUdevResourceDesc phDesc;

		//context variables
		CUgreenCtx g_Context;
		CUcontext p_Context;

		//grab TPCs needed
		CUdevResource* resources = new CUdevResource[tpcs_to_add.size()];
		for (unsigned int i = 0; i < tpcs_to_add.size(); i++){

			if (tpcs_to_add[i] > granted_TPCs){

				print_module::print(std::cerr, "ERROR: Task ", get_index(), " requested TPC ", tpcs_to_add[i], " but only has ", granted_TPCs, " TPCs.\n skipping TPC request.....\n");

			}

			else{

				resources[i] = our_TPCs[tpcs_to_add[i]];

			}

		}
			
		//make mask
		CUDA_SAFE_CALL(cuDevResourceGenerateDesc(&phDesc, resources, tpcs_to_add.size()));

		//set the mask
		CUDA_SAFE_CALL(cuGreenCtxCreate(&g_Context, phDesc, 0, CU_GREEN_CTX_DEFAULT_STREAM));

		//make the context
		CUDA_SAFE_CALL(cuCtxFromGreenCtx(&p_Context, g_Context));

		//free the inut resources
		delete[] resources;

		return p_Context;

	}

#endif
