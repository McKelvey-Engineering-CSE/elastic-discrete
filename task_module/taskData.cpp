#include "taskData.h"	

TaskData::TaskData(){}

TaskData::TaskData(bool free_resources){

	//open the 3 message queues which facilitate the process of giving CPUs or GPUs to other tasks

	//queue 1 is used to signal that a task is giving up CPUs
	if ((queue_one = msgget(98173, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 1.\n");
		kill(0, SIGTERM);

	}

	//queue 2 is used to signal that a task is giving CPUs to another task
	if ((queue_two = msgget(98174, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 2.\n");
		kill(0, SIGTERM);

	}

	//queue 3 is used to signal that a task should give up the resources contained in the mask to the task specified
	if ((queue_three = msgget(98175, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 3.\n");
		kill(0, SIGTERM);

	}

	index = counter++;

}

static std::vector<std::pair<int,int>> computeModeResources(double CpA, double L_A,
   										                    double CpB, double L_B,
        									                double T){

    // bounds on mA:
    int mA_min = (CpA == 0 ? 1 : std::ceil((CpA - L_A) / (T - L_A)));
    int mA_max = (CpA == 0 ? 1 : std::ceil((CpA - L_A) / 1));

	if (mA_max > NUMCPUS)
		mA_max = NUMCPUS;

    std::vector<std::pair<int,int>> result;
    
    for (int mA = mA_min; mA <= mA_max; ++mA) {
        
        // actual A-phase finish time
        double tA = ((CpA - L_A) / mA) + L_A;
        
        if (tA < 0) continue;
        
        double tB_window = T - tA;
        if (tB_window <= L_B) continue;

        // cores needed for B
        int mB = (CpB == 0 ? 0 : std::ceil((CpB - L_B) / (tB_window - L_B)));
            
        bool add = true;
        
        for (auto res : result){
            if (res.second == mB && res.first <= mA)
                add = false;
        }
        
        if (add)
            result.emplace_back(mA, mB);
            
        
    }

	if (result.empty()){
		print_module::print(std::cerr, "Error: No valid mode found for task with parameters ", CpA, " ", L_A, " ", CpB, " ", L_B, " ", T, "\n");
		exit(-1);
	}
    
    return result;
    
}

TaskData::TaskData(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, 
														timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_, bool safe) : 	
																													
																													index(counter++), changeable(true), 
																													can_reschedule(false), num_adaptations(0),  
																													elasticity(elasticity_), num_modes(num_modes_), 
																													max_utilization(0), max_CPUs(0), min_CPUs(NUMCPUS), 
																													max_GPUs(0), min_GPUs(NUMCPUS),  
																													CPUs_gained(0),  
																													practical_max_CPUs(max_CPUs), current_lowest_CPU(-1), 
																													percentage_workload(1.0), current_period({0,0}), 
																													current_work({0,0}), current_span({0,0}), 
																													current_utilization(0.0), current_CPUs(0), previous_CPUs(0), 
																													permanent_CPU(-1), current_mode(0), max_work({0,0}){
	
	if (num_modes > MAXMODES){

		print_module::print(std::cerr, "ERROR: No task can have more than ", MAXMODES,  " modes.\n");
		kill(0, SIGTERM);
	
	}		

	int mode_options = 0;
	int next_position = 0;

	//determine resources
	for (int i = 0; i < num_modes; i++){

		//fetch the mode parameters
		timespec work_l = work_[i];
		timespec span_l = span_[i];
		timespec period_l = period_[i];
		timespec GPU_work_l = gpu_work_[i];
		timespec GPU_span_l = gpu_span_[i];
		timespec GPU_period_l = gpu_period_[i];

		//fetch each mode parameter as a long
		double work_long = get_timespec_in_ns(work_l);
		double span_long = get_timespec_in_ns(span_l);
		double period_long = get_timespec_in_ns(period_l);
		double GPU_work_long = get_timespec_in_ns(GPU_work_l);
		double GPU_span_long = get_timespec_in_ns(GPU_span_l);

		//pass to the computeModeResources function
		auto resources = computeModeResources(work_long, span_long, GPU_work_long, GPU_span_long, period_long);
		mode_options += resources.size();

		//loop over the resources and store them in the table
		for (auto res : resources){

			//store the CPU and GPU resources
			CPUs[next_position] = res.first;
			GPUs[next_position] = res.second;

			//map the mode
			mode_map[next_position] = i;

			//update the parameters
			if (CPUs[next_position] > max_CPUs)
				max_CPUs = CPUs[next_position];

			if (CPUs[next_position] < min_CPUs)
				min_CPUs = CPUs[next_position];

			if (GPUs[next_position] > max_GPUs)
				max_GPUs = GPUs[next_position];

			if (GPUs[next_position] < min_GPUs)
				min_GPUs = GPUs[next_position];

			//set the params for this mode
			work[next_position] = work_l;
			span[next_position] = span_l;
			period[next_position] = period_l;
			GPU_work[next_position] = GPU_work_l;
			GPU_span[next_position] = GPU_span_l;
			GPU_period[next_position] = GPU_period_l;

			//note which mode this came from
			owning_modes[next_position] = i;

			//update max utilization
			if (((work_l / period_l) + (GPU_work_l / period_l)) > max_utilization)
				max_utilization = ((work_l / period_l) + (GPU_work_l / period_l));

			if (++next_position >= MAXMODES){

				print_module::print(std::cerr, "ERROR: No task can have more than ", MAXMODES, " modes.\n");
				kill(0, SIGTERM);
			
			}

		}

		if (work[i] > max_work)
			max_work = work[i];

		if (GPU_work[i] != timespec({0, 0}) &&  GPU_span[i] != timespec({0, 0}))
			is_pure_cpu_task = false;

	}

	print_module::print(std::cout, "Task ", index, " has ", num_modes, " modes:\n");
	for (int i = 0; i < num_modes; i++)
		print_module::print(std::cout, work_[i], " ", span_[i], " ", period_[i], " ", gpu_work_[i], " ", gpu_span_[i], "\n");
	print_module::print(std::cout, "\n");
	
	modes_originally_passed = num_modes;

	num_modes = mode_options;
	number_of_modes = mode_options;

	//check if we should be making this task safe from false negatives
	if (safe){

		//look through the lowest mode and for each possible state 
		//select a mode which represents the lowest possible state
		//and look through all the other mode_options, and count the
		//number of mode options which are stricly larger
		//than the current mode option. Ensure at least one candidate mode is present
		//from each mode transition possible

		int best_base_mode = 0;
		int best_base_mode_seen = 0;

		for (int i = 0; i < num_modes; i++){

			int current_cpu = CPUs[i];
			int current_gpu = GPUs[i];

			int distinct_modes_seen = 0;
			int seen_modes_utilizations[modes_originally_passed] = {0};

			//set the first seen mode as itself
			seen_modes_utilizations[owning_modes[i]] = 1;

			for (int j = 0; j < num_modes; j++){

				if (i != j){

					if (CPUs[j] >= current_cpu && GPUs[j] >= current_gpu){

						distinct_modes_seen++;
						seen_modes_utilizations[owning_modes[j]] += 1;

					}

				}

			}

			if (best_base_mode_seen < distinct_modes_seen){

				bool candidate_mode = true;
				for (int i = 0; i < modes_originally_passed; i++){

					if (seen_modes_utilizations[i] == 0)
						candidate_mode = false;

				}

				if (candidate_mode){

					best_base_mode = i;
					best_base_mode_seen = distinct_modes_seen;

				}

			}

		}


		//if we have a best base mode set, we can use that as
		//the standard by which we forcibly inflate all other 
		//modes
		if (best_base_mode_seen > 0){

			for (int i = 0; i < num_modes; i++){

				if (CPUs[i] < CPUs[best_base_mode])
					CPUs[i] = CPUs[best_base_mode];

				if (GPUs[i] < GPUs[best_base_mode])
					GPUs[i] = GPUs[best_base_mode];

			}

			print_module::print(std::cout, "Task ", index, " has been forced to inflate to mode ", best_base_mode, "\n");

		}

		else {

			print_module::print(std::cout, "Task ", index, " has no base mode to inflate to. The system cannot be safely scheduled.\n");
			exit(-1);

		}


	}

	//loop over all modes, and compare the allocated processor A
	//and processor B to all other modes in the task. If the task
	//has any mode where it gains cpus and loses gpus compared to 
	//any other mode in the task or vice versa, then the task is
	//combinatorially elastic
	for (int i = 0; i < num_modes; i++){

		for (int j = 0; j < num_modes; j++){

			if (i != j){

				if (CPUs[i] > CPUs[j] && GPUs[i] < GPUs[j]){

					combinatorially_elastic = true;
					break;

				}

				if (CPUs[i] < CPUs[j] && GPUs[i] > GPUs[j]){

					combinatorially_elastic = true;
					break;

				}

			}

		}

	}

	current_CPUs = min_CPUs;
	current_GPUs = min_GPUs;

	//clear the tables so I can actually read them when needed
	for (int i  = 0; i < MAXTASKS + 1; i++){

		for (int j = 0; j < NUMGPUS + 1; j++){

			gpus_granted_from_other_tasks[i][j] = -1;

		}
	}

	for (int i  = 0; i < MAXTASKS + 1; i++){

		for (int j = 0; j < NUMCPUS + 1; j++){

			cpus_granted_from_other_tasks[i][j] = -1;

		}
	}

	//open the 3 message queues which facilitate the process of giving CPUs or GPUs to other tasks

	//queue 1 is used to signal that a task is giving up CPUs
	if ((queue_one = msgget(98173, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 1.\n");
		kill(0, SIGTERM);

	}

	//queue 2 is used to signal that a task is giving CPUs to another task
	if ((queue_two = msgget(98174, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 2.\n");
		kill(0, SIGTERM);

	}

	//queue 3 is used to signal that a task should give up the resources contained in the mask to the task specified
	if ((queue_three = msgget(98175, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 3.\n");
		kill(0, SIGTERM);

	}
}

TaskData::~TaskData(){
}

int TaskData::counter = 0;	        

int TaskData::get_index(){
	return index;
}

int TaskData::get_number_of_modes(){
	return number_of_modes;
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

int TaskData::get_real_mode(int mode){
	return mode_map[mode];
}

void TaskData::reset_mode_to_previous(){

	current_mode = previous_mode;
	current_work = work[current_mode];
	current_span = span[current_mode];
	current_period = period[current_mode];
	current_utilization = current_work / current_period;
	percentage_workload = current_work / max_work;
	current_CPUs = previous_CPUs;
	current_GPUs = previous_GPUs;

}

void TaskData::set_current_mode(int new_mode, bool disable)
{
	if (new_mode >= 0 && new_mode < num_modes){

		//stash old mode
		previous_mode = current_mode;

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

		//set the current mode notation to something the task actually can use
		real_current_mode = get_real_mode(current_mode);

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

bool TaskData::pure_cpu_task(){
	return is_pure_cpu_task;
}

int TaskData::get_original_modes_passed(){
	return modes_originally_passed;
}

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

bool TaskData::check_mode_transition(){
	return mode_transitioned;
}

void TaskData::set_mode_transition(bool state){
	mode_transitioned = state;
}

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
		//skip the current bit
		msb--;
		test_bit >>= 1;

		while ((CPU_mask & test_bit) == 0) {
			msb--;
			test_bit >>= 1;
		}
	}

    // Clear the MSB
    CPU_mask ^= test_bit;

    return msb;
}

int TaskData::pop_back_gpu(){
    // Handle empty vector case
    if (TPC_mask == 0) {
        return -1;
    }

    int msb = 127;  // Start from highest possible bit
    __uint128_t test_bit = (__uint128_t)1 << 127;

    // Find the most significant 1 bit
    while ((TPC_mask & test_bit) == 0) {
        msb--;
        test_bit >>= 1;
    }

    // Clear the MSB
    TPC_mask ^= test_bit;

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

int TaskData::push_back_gpu(int value){
    // Check if value is in valid range
    if (value < 0 || value > 127) {
        return false;
    }
    
    // Check if bit is already set
    __uint128_t bit = (__uint128_t)1 << value;
    if (TPC_mask & bit) {
        return false;
    }
    
    // Set the bit
    TPC_mask |= bit;
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

		if (TPC_mask & ((__uint128_t)1 << i)) {

			TPCS_owned_by_task.push_back(i);

		}
	}

	return TPCS_owned_by_task;

}

//retrieve the number of CPUs or GPUs we have been given	
std::vector<std::pair<int, std::vector<int>>> TaskData::get_cpus_granted_from_other_tasks(){

	std::vector<std::pair<int, std::vector<int>>> returning_cpus_granted_from_other_tasks;

	//stupid conversion to make the vectors
	for (int i = 0; i < MAXTASKS + 1; i++){
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
	for (int i = 0; i < MAXTASKS + 1; i++){
		if (gpus_granted_from_other_tasks[i][0] != -1){

			auto current = std::make_pair(i, std::vector<int>());

			for (int j = 1; j < gpus_granted_from_other_tasks[i][0] + 1; j++)
				current.second.push_back(gpus_granted_from_other_tasks[i][j]);

			returning_gpus_granted_from_other_tasks.push_back(current);
		}

	}

	return returning_gpus_granted_from_other_tasks;

}

//give CPUs or GPUs to another task

/*****************************************************************************************


If a task is giving up CPUs, keep track of what tasks it gives to in a table.
When all tasks have been processed, send a message from the scheduler of type -1 to 
the first queue with the message id set to the task index in question. 

All other tasks will read from this queue when scheduler signals that it is time to
transition, and any process which gives up cpus will send a message back into that same
queue with the message id set to the task index in question.

2 System V Message Queues are used to facilitate this process.

Queue 1: Used to signal that a task is giving up CPUs. The message type is the task index

Queue 2: Id determines process target. Message contents are the CPUs or GPUs to be given by the processor number

Queue 3: Used to to signal that a task should give up the resources contained in the mask to the task specified

struct queue_one_message {
    long int    mtype;
	__uint128_t tasks_giving_processors;
}

struct queue_two_message {
	long int    mtype;
	long int   giving_task;
	long int   processor_type;
	__uint128_t processors;
}

struct queue_three_message {
	long int    mtype;
	long int   task_index;
	long int   processor_type;
	long int   processor_ct;
}

Structure:

If Time to Reschedule:

	Read From queue 1 for message of id == task_index

		- If message type == -1 : global_variable tasks_to_listen_for = message.tasks_giving_processors

	If all tasks_to_listen_for == 0, then we can transition

		For each read from Queue 2

			If processor_type == CPU : CPU_mask |= processors
			If processor_type == GPU : TPC_mask |= processors

		OR these values with the current masks

		Read from Queue 3

			If processor_type == CPU : CPU_mask &= ~processors
			If processor_type == GPU : TPC_mask &= ~processors

			send message to queue 2 with the processors to be given



		Transition complete


******************************************************************************************/

void TaskData::set_processors_to_send_to_other_processes(int task_to_send_to, int processor_type, int processor_ct){

	//send message into queue 3
	struct queue_three_message message;

	message.mtype = get_index() + 1;
	message.task_index = task_to_send_to + 1;
	message.processor_type = processor_type;
	message.processor_ct = processor_ct;

	//std::cerr<< "Telling task " << get_index() << " to give " << processor_ct << " processors to task " << task_to_send_to << ".\n";

	if (msgsnd(queue_three, &message, sizeof(message) - sizeof(long), 0) == -1){

		print_module::print(std::cerr, "Error: Failed to send a message to queue 3: failed with: ", strerror(errno), " \n");
		kill(0, SIGTERM);

	}
	
}

void TaskData::set_tasks_to_wait_on(__uint128_t task_mask){	

	//send message into queue 1
	struct queue_one_message message;

	message.mtype = get_index() + 1;
	message.tasks_giving_processors = task_mask;

	if (msgsnd(queue_one, &message, sizeof(message) - sizeof(long), 0) == -1){

		print_module::print(std::cerr, "Error: Failed to send message to queue 1.\n");
		kill(0, SIGTERM);

	}
	
}

void TaskData::start_transition(){

	//read from queue 1 
	struct queue_one_message message;

	if (msgrcv(queue_one, &message, sizeof(message) - sizeof(long), get_index() + 1, IPC_NOWAIT) != -1){
		
		tasks_giving_processors = message.tasks_giving_processors;

		std::cerr << "I am task " << get_index() << " and I am waiting on " << (unsigned long long) tasks_giving_processors << " tasks.\n";

	}

}

//fetch any processors that have been given to us
//and returns true when all the resources have been given
//to us from other processes: NON BLOCKING
bool TaskData::get_processors_granted_from_other_tasks(){

	//read from queue 2, until we have no messages available
	struct queue_two_message message;

	while (msgrcv(queue_two, &message, sizeof(message) - sizeof(long), get_index() + 1, IPC_NOWAIT) != -1){

		//if the message is for CPUs
		if (message.processor_type == 0)
			processors_A_received |= message.processors;

		//if the message is for GPUs
		else if (message.processor_type == 1)
			processors_B_received |= message.processors;

		tasks_giving_processors &= ~((__uint128_t)(1) << message.giving_task);

	}

	//std::cerr<< "Task " << get_index() << " has received " << (unsigned long long) processors_A_received << " CPUs and " << (unsigned long long) processors_B_received << " GPUs.\n";

	//if (tasks_giving_processors != 0)
		//std::cerr<< "Task " << get_index() << " is still waiting on " << (unsigned long long) tasks_giving_processors << " tasks.\n";

	return tasks_giving_processors == 0;

}

void TaskData::set_cpus_to_send_to_other_processes(std::pair<int, int> entry){

	//send message into queue 2
	struct queue_two_message message;

	message.mtype = entry.first;
	message.giving_task = get_index();
	message.processor_type = 0;
	message.processors = 0;

	//always grabs from the back of the vector
	for (int i = 0; i < entry.second; i++){

		int cpus_selected_to_send = pop_back_cpu();

		message.processors |= ((__uint128_t)1 << cpus_selected_to_send);
	}

	if (msgsnd(queue_two, &message, sizeof(message) - sizeof(long), 0) == -1){

		print_module::print(std::cerr, "Error: Failed to send message to queue 2.\n");
		kill(0, SIGTERM);

	}

}

void TaskData::set_gpus_to_send_to_other_processes(std::pair<int, int> entry){

	//send message into queue 2
	struct queue_two_message message;

	message.mtype = entry.first;
	message.giving_task = get_index();
	message.processor_type = 1;
	message.processors = 0;

	//always grabs from the back of the vector
	for (int i = 0; i < entry.second; i++){

		int gpus_selected_to_send = pop_back_gpu();

		//std::cerr << "Task " << get_index() << " is sending GPU " << gpus_selected_to_send << " to task " << entry.first << ".\n";

		message.processors |= ((__uint128_t)1 << gpus_selected_to_send);

	}

	if (msgsnd(queue_two, &message, sizeof(message) - sizeof(long), 0) == -1){

		print_module::print(std::cerr, "Error: Failed to send message to queue 2.\n");
		kill(0, SIGTERM);

	}

}	

void TaskData::acquire_all_processors(){
	
	//if we have been given processors, then we can acquire them
	CPU_mask |= processors_A_received;
	TPC_mask |= processors_B_received;

	//clear the variables
	processors_A_received = 0;
	processors_B_received = 0;
	tasks_giving_processors = 0;

}

void TaskData::give_processors_to_other_tasks(){

	//read from queue 3 until we cant
	struct queue_three_message message;

	while (msgrcv(queue_three, &message, sizeof(message) - sizeof(long), get_index() + 1, IPC_NOWAIT) != -1){

		//if the message is for CPUs
		if (message.processor_type == 0){

			set_cpus_to_send_to_other_processes(std::make_pair(message.task_index, message.processor_ct));

			//std::cerr<< "Task " << get_index() << " is giving " << message.processor_ct << " CPUs to task " << message.task_index << ".\n";

		}

		//if the message is for GPUs
		else if (message.processor_type == 1){

			set_gpus_to_send_to_other_processes(std::make_pair(message.task_index, message.processor_ct));

			//std::cerr<< "Task " << get_index() << " is giving " << message.processor_ct << " GPUs to task " << message.task_index << ".\n";

		}

	}

	//std::cerr<< "Task " << get_index() << " finished giving processors.\n";

}

//make a function which clears these vectors like they are cleared in the constructor
void TaskData::clear_cpus_granted_from_other_tasks(){

	for (size_t i = 0; i < MAXTASKS + 1; i++)
		cpus_granted_from_other_tasks[i][0] = -1;

}

void TaskData::clear_gpus_granted_from_other_tasks(){

	for (size_t i = 0; i < MAXTASKS + 1; i++)
		gpus_granted_from_other_tasks[i][0] = -1;

}

__uint128_t TaskData::get_cpu_mask() {
	
	return CPU_mask;

}

__uint128_t TaskData::get_gpu_mask() {
	
	return TPC_mask;

}

bool TaskData::is_combinatorially_elastic(){
	return combinatorially_elastic;
}

void TaskData::set_cooperative(bool state){

	cooperative_bool = state;

}

bool TaskData::cooperative(){
	
	return cooperative_bool;

}