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

	//if we are compiling using NVCC on a CUDA-enabled machine, update this param
	#ifdef __NVCC__

		CUdevResource initial_resources;

		//init device driver
		CUDA_SAFE_CALL(cuInit(0));

		//fill the initial descriptor
		CUDA_SAFE_CALL(cuDeviceGetDevResource(0, &initial_resources, CU_DEV_RESOURCE_TYPE_SM));

		//get the individual TPC slices
		CUDA_SAFE_CALL(cuDevSmResourceSplitByCount(total_TPCs, &num_TPCs, &initial_resources, NULL, CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));

		NUMGPUS = num_TPCs;

	#endif

	//make the GPU related stuff
	active_gpus = new int[NUMGPUS + 1];
	passive_gpus = new int[NUMGPUS + 1];

	for (int i = 0; i < MAXTASKS; i++){

		transfer_GPU[i] = new bool[NUMGPUS + 1];
		receive_GPU[i] = new bool[NUMGPUS + 1];

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
		ts_diff(GPU_work[i], GPU_span[i], numerator);
		ts_diff(GPU_period[i], GPU_span[i], denominator);

		if (GPU_work[i] != timespec({0, 0}) &&  GPU_span[i] != timespec({0, 0}) && GPU_period[i] != timespec({0, 0})){

			GPUs[i] = (int)ceil(numerator / denominator);

			is_pure_cpu_task = false;

		}

		else{

			GPUs[i] = 0;

		}

		if (GPUs[i] > max_GPUs)
			max_GPUs = GPUs[i];

		if (GPUs[i] < min_GPUs)
			min_GPUs = GPUs[i];

	}

	current_CPUs = min_CPUs;
	current_GPUs = min_GPUs;

	for (int i = 0; i < MAXTASKS; i++){
		
		//set shared to 0
		give_CPU[i] = 0;
		give_GPU[i] = 0;

		//set CPU to 0
		for (int j = 1; j <= NUMCPUS; j++){

			transfer_CPU[i][j] = false;
			receive_CPU[i][j] = false;
			active_cpus[i] = false;
			passive_cpus[i] = false;

		}

		//set GPU to 0
		for (int j = 1; j <= NUMGPUS; j++){

			transfer_GPU[i][j] = false;
			receive_GPU[i][j] = false;
			active_gpus[i] = false;
			passive_gpus[i] = false;

		}
	}
}

TaskData::~TaskData(){

	//clear the GPU related stuff
	delete[] active_gpus;
	delete[] passive_gpus;

	for (int i = 0; i < MAXTASKS; i++){
		delete[] transfer_GPU[i];
		delete[] receive_GPU[i];
	}

}

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

int TaskData::get_GPUs_gained(){

	return GPUs_gained;

}

void TaskData::set_GPUs_gained(int new_GPUs_gained){

	GPUs_gained = new_GPUs_gained;

}

int TaskData::get_previous_CPUs(){

	return previous_CPUs;

}

void TaskData::set_previous_CPUs(int new_prev){

	previous_CPUs = new_prev;

}

void TaskData::update_give(int index, int value){

	give_CPU[index] = value;

}

void TaskData::update_gpu_give(int index, int value){

	give_GPU[index] = value;

}

int TaskData::gives(int index){

	return give_CPU[index];

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

int TaskData::get_previous_GPUs(){

	return previous_GPUs;

}

void TaskData::set_previous_GPUs(int new_prev){

	previous_GPUs = new_prev;

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

		//update CPU parameters
		current_mode = new_mode;
		current_work = work[current_mode];
		current_span = span[current_mode];
		current_period = period[current_mode];
		current_utilization = current_work / current_period;
		percentage_workload = current_work / max_work;
		previous_CPUs = current_CPUs;
		current_CPUs = CPUs[current_mode];
		
		//update GPU parameters
		previous_GPUs = current_GPUs;
		current_GPUs = GPUs[current_mode];

		//update the changeable flag
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

	return transfer_CPU[task][CPU];

}
	
void TaskData::set_transfer(int task, int CPU, bool value){

	transfer_CPU[task][CPU] = value;

}

bool TaskData::receives(int task, int CPU){

	return receive_CPU[task][CPU];

}

void TaskData::set_receive(int task, int CPU, bool value){

	receive_CPU[task][CPU] = value;

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

int TaskData::get_total_TPC_count(){

	return NUMGPUS;

}

bool TaskData::pure_cpu_task(){

	return is_pure_cpu_task;

}

std::vector<int> TaskData::retract_GPUs(int value){

	//loop over our TPC mask and if a bit is set to 1, add it to the vector and set it to 0 in the new mask
	std::vector<int> TPCs_to_retract;

	for (int i = 0; i < 128; i++){

		if (TPC_mask & (((__uint128_t)1 << i) >> 1) && value > 0){

			TPCs_to_retract.push_back(i);
			TPC_mask &= ~(((__uint128_t)1 << i) >> 1);
			value--;

		}
	}

	return TPCs_to_retract;

}

void TaskData::gifted_GPUs(std::vector<int> TPCs_to_grant){

	//do the opposite of the function above, for each int in the TPC_to_grant vector, set the corresponding bit to 1 in the mask
	for (unsigned int i = 0; i < TPCs_to_grant.size(); i++){

		TPC_mask |= (((__uint128_t)1 << (__uint128_t)TPCs_to_grant.at(i)) >> 1);

	}

}

//related GPU functions
#ifdef __NVCC__

	//returns the mask for the given task
	__uint128_t TaskData::get_TPC_mask(){

		return TPC_mask;

	}

	//quick function to create a SM stream
	cudaStream_t TaskData::create_partitioned_stream(int TPCs){

		cudaStream_t stream;
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

		//if TPCs is a positive number, check that we have enough TPCs to grant 
		//and make a new mask to apply
		if (TPCs > 0){

			//check that we have enough TPCs to grant
			if (TPCs > __builtin_popcount(TPC_mask)){

				print_module::print(std::cerr, "Error: Task ", get_index(), " was told to grant ", TPCs, " TPCs, but only has ", __builtin_popcount(TPC_mask), " to grant. Ignoring.\n");
				return stream;

			}

			//make a new mask
			__uint128_t TPC_mask_new = 0;

			for (int i = 0; i < TPCs; i++){

				TPC_mask_new |= (((__uint128_t)1 << (__uint128_t)i) >> 1);

			}

			//apply mask
			libsmctrl_set_stream_mask(stream, TPC_mask_new);

			return stream;

		}

		//apply mask
		libsmctrl_set_stream_mask(stream, TPC_mask);

		return stream;
	}

	//quick function to update an existing stream
	void TaskData::update_partitioned_stream(cudaStream_t& stream, int TPCs){

		//if TPCs is a positive number, check that we have enough TPCs to grant 
		//and make a new mask to apply
		if (TPCs > 0){

			//check that we have enough TPCs to grant
			if (TPCs > __builtin_popcount(TPC_mask)){

				print_module::print(std::cerr, "Error: Task ", get_index(), " was told to grant ", TPCs, " TPCs, but only has ", __builtin_popcount(TPC_mask), " to grant. Ignoring.\n");
				return;

			}

			//make a new mask
			__uint128_t TPC_mask_new = 0;

			for (int i = 0; i < TPCs; i++){

				TPC_mask_new |= (((__uint128_t)1 << (__uint128_t)i) >> 1);

			}

			//apply mask
			libsmctrl_set_stream_mask(stream, TPC_mask_new);

			return;

		}

		//apply mask
		libsmctrl_set_stream_mask(stream, TPC_mask);

		return;
	}

#endif


//reworking all the CPU and GPU handoff functions
//NOTE: all return functions will work from the 
//highest CPU/SM unit we have down until we run
//out of CPUs/SMs to return
void TaskData::set_CPUs_change(int num_cpus_to_return){
	cpus_to_return = num_cpus_to_return;
}

void TaskData::set_GPUs_change(int num_gpus_to_return){
	gpus_to_return = num_gpus_to_return;
}

int TaskData::get_CPUs_change(){
	return cpus_to_return;
}

int TaskData::get_GPUs_change(){
	return gpus_to_return;
}

//function to check if this task has transitioned
//to a new mode yet
bool TaskData::check_mode_transition(){
	return mode_transitioned;
}

void TaskData::set_mode_transition(bool state){
	mode_transitioned = state;
}

//functions to work with static vector of CPU indices
int TaskData::pop_back_cpu(){

    // Handle empty vector case
    if (CPU_mask == 0) {
        return -1;
    }

    int msb = 127;  // Start from highest possible bit
    __uint128_t test_bit = (__uint128_t)1 << 127;

    // Find the most significant 1 bit
    while ((CPU_mask & test_bit) == 0) {
        msb--;
        test_bit >>= 1;
    }

	//check if it's our permanent
	if (msb == get_permanent_CPU()){
		 while ((CPU_mask & test_bit) == 0) {
			msb--;
			test_bit >>= 1;
		}
	}

    // Clear the MSB
    CPU_mask ^= test_bit;

    return msb;

}

int TaskData::push_back_cpu(int value){

    // Check if value is in valid range
    if (value < 0 || value > 127) {
        return false;
    }
    
    // Check if bit is already set
    __uint128_t bit = (__uint128_t)1 << value;
    if (CPU_mask & bit) {
        return false;
    }
    
    // Set the bit
    CPU_mask |= bit;
    return true;
}

std::vector<int> TaskData::get_cpu_owned_by_process(){
	
    std::vector<int> result;

    result.reserve(128);
    
    for (int i = 0; i < 128; i++) {
        if (CPU_mask & ((__uint128_t)1 << i)) {

			//do not allow our permanent CPU to be returned as
			//a cpu we can pass or keep
			if (i != get_permanent_CPU())
            	result.push_back(i);
        }
    }

    return result;

}

std::vector<int> TaskData::get_gpu_owned_by_process(){

	//loop over our TPC mask and if a bit is set to 1, add it to the vector
	std::vector<int> TPCS_owned_by_task;

	for (int i = 0; i < 128; i++){

		if (TPC_mask & (((__uint128_t)1 << i) >> 1)){

			TPCS_owned_by_task.push_back(i);

		}
	}

	return TPCS_owned_by_task;

}

//retrieve the number of CPUs or GPUs we have been given	
std::vector<std::pair<int, std::vector<int>>> TaskData::get_cpus_granted_from_other_tasks(){

	std::vector<std::pair<int, std::vector<int>>> returning_cpus_granted_from_other_tasks;

	//stupid conversion to make the vectors
	for (int i = 0; i < MAXTASKS; i++){
		if (cpus_granted_from_other_tasks[i][0] != -1){

			auto current = std::make_pair(i, std::vector<int>());

			for (int j = 1; j < cpus_granted_from_other_tasks[i][0] + 1; j++)
				current.second.push_back(cpus_granted_from_other_tasks[i][j]);

			returning_cpus_granted_from_other_tasks.push_back(current);

		}

	}

	return returning_cpus_granted_from_other_tasks;

}

std::vector<std::pair<int, std::vector<int>>> TaskData::get_gpus_granted_from_other_tasks(){

	std::vector<std::pair<int, std::vector<int>>> returning_gpus_granted_from_other_tasks;

	//stupid conversion to make the vectors
	for (int i = 0; i < MAXTASKS; i++){
		if (gpus_granted_from_other_tasks[i][0] != -1){

			auto current = std::make_pair(i, std::vector<int>());

			for (int j = 1; j < cpus_granted_from_other_tasks[i][0] + 1; j++)
				current.second.push_back(cpus_granted_from_other_tasks[i][j]);

			returning_gpus_granted_from_other_tasks.push_back(current);
		}

	}

	return returning_gpus_granted_from_other_tasks;

}

//give CPUs or GPUs to another task
void TaskData::set_cpus_granted_from_other_tasks(std::pair<int, std::vector<int>> entry){

	for (int i = 0; i < entry.second.size(); i++)
		cpus_granted_from_other_tasks[entry.first][i+1] = entry.second.at(i);

	cpus_granted_from_other_tasks[entry.first][0] = entry.second.size();

}

void TaskData::set_gpus_granted_from_other_tasks(std::pair<int, std::vector<int>> entry){

	for (int i = 0; i < entry.second.size(); i++)
		gpus_granted_from_other_tasks[entry.first][i+1] = entry.second.at(i);

	gpus_granted_from_other_tasks[entry.first][0] = entry.second.size();

}	

//make a function which clears these vectors like they are cleared in the constructor
void TaskData::clear_cpus_granted_from_other_tasks(){

	for (size_t i = 0; i < MAXTASKS; i++)
		cpus_granted_from_other_tasks[i][0] = -1;

}

void TaskData::clear_gpus_granted_from_other_tasks(){

	for (size_t i = 0; i < MAXTASKS; i++)
		gpus_granted_from_other_tasks[i][0] = -1;

}

__uint128_t TaskData::get_cpu_mask() {
	
	return CPU_mask;

}