// Argument: the name of the taskset/schedule file:

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <sys/wait.h>
#include <list>
#include <thread>
#include <cerrno>

#include "process_barrier.h"
#include "timespec_functions.h"
#include "scheduler.h"
#include "print_module.h"
#include "memory_allocator.h"

/************************************************************************************
Globals
*************************************************************************************/

timespec current_time,start_time,run_time,end_time;
timespec cur_time;
timespec vdeadline={0,0};
timespec zero={0,0};
int ret_val;
int get_val=0;
struct itimerspec disarming_its, virtual_dl_timer_its;
struct sigevent sev;
int ret;

bool FPTAS = false;

// Define the name of the barrier used for synchronizing tasks after creation
const std::string barrier_name = "BAR";
const std::string barrier_name2 = "BAR_2";

bool needs_scheduling = false;
Scheduler * scheduler;
pid_t process_group;

//BOOLEAN VALUE FOR WHETHER WE INITIALLY INCLUDE LAST TASK IN THE SCHEDULE
bool add_last_task = true;
TaskData * last_task;

//bool which controls whether or not we are running with explicit syncronization
bool explicit_sync = false;

//bool to control whether or not to force the scheduler to be safe
bool safe_calculation = false;

//bool to control whether or not the scheduler is being run in simulated task mode
bool simulated_task_mode = false;

/************************************************************************************
Objects
*************************************************************************************/

enum rt_gomp_clustering_launcher_error_codes
{ 
	RT_GOMP_CLUSTERING_LAUNCHER_SUCCESS,
	RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR,
	RT_GOMP_CLUSTERING_LAUNCHER_FILE_PARSE_ERROR,
	RT_GOMP_CLUSTERING_LAUNCHER_UNSCHEDULABLE_ERROR,
	RT_GOMP_CLUSTERING_LAUNCHER_FORK_EXECV_ERROR,
	RT_GOMP_CLUSTERING_LAUNCHER_BARRIER_INITIALIZATION_ERROR,
	RT_GOMP_CLUSTERING_LAUNCHER_ARGUMENT_ERROR
};

/************************************************************************************
Functions
*************************************************************************************/

void simulated_scheduler()
{
	
	scheduler->do_schedule(NUMCPUS - 1, true);

}

//function of pure vanity courtesy of claude
static std::string convertToRanges(const std::string& input) {
    std::vector<int> numbers;
    std::istringstream iss(input);
    std::string token;
    
    // Parse input string into vector of integers
    while (std::getline(iss, token, ',')) {
        // Trim leading and trailing whitespace
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        
        // Skip empty tokens (handles multiple commas)
        if (!token.empty()) {
            numbers.push_back(std::stoi(token));
        }
    }
    
    if (numbers.empty()) return "";

    // Sort the numbers in ascending order
    std::sort(numbers.begin(), numbers.end());

    std::ostringstream result;
    int start = numbers[0], prev = numbers[0];
    
    for (size_t i = 1; i <= numbers.size(); ++i) {
        if (i == numbers.size() || numbers[i] != prev + 1) {
            if (start == prev) {
                result << start;
            } else {
                result << start << "-" << prev;
            }
            
            if (i < numbers.size()) {
                result << ",";
                start = numbers[i];
            }
        }
        if (i < numbers.size()) prev = numbers[i];
    }
    
    return result.str();
}

void simulated_task_start(int task_index, timespec * current_period, timespec * current_work, int * current_mode, timespec * deadline, int * percentile, __uint128_t * current_cpu_mask, __uint128_t * current_gpu_mask, Schedule* schedule){

	//Collect our first schedule and set it ups
	*current_work = schedule->get_task(task_index)->get_current_work();
	*current_period = schedule->get_task(task_index)->get_current_period();
	
	//print task information
	std::ostringstream task_info;
	std::string task_header = "|  Task " + std::to_string(task_index) + " (PID: " + std::to_string(getpid()) + ") |\n";
	size_t blank_size = task_header.size();

	print_module::buffered_print(task_info, "\n", std::string(blank_size - 1, '='), "\n");
	print_module::buffered_print(task_info, task_header);
	print_module::buffered_print(task_info, std::string(blank_size - 1, '='), "\n\n");

	//cpu info
	print_module::buffered_print(task_info, "CPU Metrics: \n");
	print_module::buffered_print(task_info, "	- Lowest CPU: ", schedule->get_task(task_index)->get_current_lowest_CPU(), "\n");
	print_module::buffered_print(task_info, "	- Current CPUs: ", schedule->get_task(task_index)->get_current_CPUs(), "\n");
	print_module::buffered_print(task_info, "	- Minimum CPUs: ", schedule->get_task(task_index)->get_min_CPUs(), "\n");
	print_module::buffered_print(task_info, "	- Maximum CPUs: ", schedule->get_task(task_index)->get_max_CPUs(), "\n");

	//gpu info
	print_module::buffered_print(task_info, "GPU Metrics: \n");
	print_module::buffered_print(task_info, "	- Lowest GPU: ", schedule->get_task(task_index)->get_current_lowest_GPU(), "\n");
	print_module::buffered_print(task_info, "	- Current GPUs: ", schedule->get_task(task_index)->get_current_GPUs(), "\n");
	print_module::buffered_print(task_info, "	- Minimum GPUs: ", schedule->get_task(task_index)->get_min_GPUs(), "\n");
	print_module::buffered_print(task_info, "	- Maximum GPUs: ", schedule->get_task(task_index)->get_max_GPUs(), "\n");

	//timing info
	print_module::buffered_print(task_info, "Timing Metrics: \n");
	print_module::buffered_print(task_info, "	- Period s: ", schedule->get_task(task_index)->get_current_period().tv_sec , " s\n");
	print_module::buffered_print(task_info, "	- Period ns: ", schedule->get_task(task_index)->get_current_period().tv_nsec , " ns\n\n");

	//Determine which threads are active and passive. Pin, etc.
	std::string active_cpu_string = "";
	std::string passive_cpu_string = "";

	//for CPUs that we MAY have at some point (practical max)
	for (int j = 1; j < NUMCPUS; j++){

		//Our first CPU is our permanent CPU 
		if (j == schedule->get_task(task_index)->get_current_lowest_CPU()){

			schedule->get_task(task_index)->set_permanent_CPU(j);

		}

		if (j >= schedule->get_task(task_index)->get_current_lowest_CPU() && j < schedule->get_task(task_index)->get_current_lowest_CPU() + schedule->get_task(task_index)->get_current_CPUs()){

			active_cpu_string += std::to_string(j) + ", ";

			schedule->get_task(task_index)->push_back_cpu(j);

		}

		else {

			passive_cpu_string += std::to_string(j) + ", ";

		}
 
  	}

	//set the cpu mask
	*current_cpu_mask = schedule->get_task(task_index)->get_cpu_mask();
	std::cout << "Process: " << task_index <<  " Initial CPU Mask: " << (unsigned long long) *current_cpu_mask << std::endl;

	//print active vs passive CPUs
	print_module::buffered_print(task_info, "CPU Core Configuration: \n");
	print_module::buffered_print(task_info, "	- Active: ", convertToRanges(active_cpu_string), "\n");

	//command line params
	print_module::buffered_print(task_info, "Command Line Parameters: \n");
	print_module::buffered_print(task_info, "	- Args: SIMULATED SIMULATED SIMULATED", "\n\n");

	//flush all task info to terminal
	print_module::flush(std::cerr, task_info);

	//update our gpu mask
	*current_gpu_mask = schedule->get_task(task_index)->get_gpu_mask();
	
}

bool simulated_reschedule(int task_index, timespec* current_period, timespec* current_work, int* current_mode, timespec* deadline, int* percentile, __uint128_t* current_cpu_mask, __uint128_t* current_gpu_mask, Schedule* schedule){

	//Fetch the mask of tasks which we are waiting on
	//(safe to call multiple times)
	schedule->get_task(task_index)->start_transition();

	//now fetch any resources which have been granted to us
	//(will return true when our mask is empty due to all
	//resources being granted)
	if (schedule->get_task(task_index)->get_processors_granted_from_other_tasks()){
		
		//semantically a bit strange, but we can only give or
		//take from each pool, so we call this and then call our
		//giving requirements

		//finalize the transition of resources to us (we now really own them)
		schedule->get_task(task_index)->acquire_all_processors();

		//transition any resources we are supposed to be giving up
		schedule->get_task(task_index)->give_processors_to_other_tasks();

		// Set up everything to begin as scheduled.
		*current_period = schedule->get_task(task_index)->get_current_period();
		*current_work = schedule->get_task(task_index)->get_current_work();
		*current_mode = schedule->get_task(task_index)->get_current_virtual_mode();

		//update our cpu mask
		*current_cpu_mask = schedule->get_task(task_index)->get_cpu_mask();

		//update our gpu mask
		*current_gpu_mask = schedule->get_task(task_index)->get_gpu_mask();

		std::ostringstream reschedule_buffer;

		//all pretty printing crap
		#ifdef PRETTY_PRINTING

			print_module::buffered_print(reschedule_buffer, "\nTask ", task_index, " finished reschedule:\n");	

			//core A
			print_module::buffered_print(reschedule_buffer, "Core A Owned: [ ");

			std::bitset<128> cpu_mask(*current_cpu_mask);
			for (int i = 0 ; i < 128; i++)
				print_module::buffered_print(reschedule_buffer, cpu_mask.test(i) ? std::to_string(i) + " " : "");

			print_module::buffered_print(reschedule_buffer, "]\n");

			//core B
			print_module::buffered_print(reschedule_buffer, "Core B Owned: [ ");

			std::bitset<128> gpu_mask(*current_gpu_mask);
			for (int i = 0 ; i < 128; i++)
				print_module::buffered_print(reschedule_buffer, gpu_mask.test(i) ? std::to_string(i) + " " : "");

			print_module::buffered_print(reschedule_buffer, "]\n");

			print_module::flush(std::cerr, reschedule_buffer);

		#endif

		return true;

	}

	return false;

}


void scheduler_task()
{
	//stick this thread to CPU 0.	
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(0,&mask);
	pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&mask);

	//Make sure all tasks are ready. Wait at barrier.
	if ((ret_val = process_barrier::await_and_rearm_barrier(barrier_name)) != 0)
	{
		print_module::print(std::cerr, "ERROR: Barrier error for scheduling task.\n");
		kill(0, SIGTERM);
	}

	//Constantly see if we need to reschedule until time is up.
	//If we need to reschedule tasks, do it and then let them know. 
	get_time(&cur_time);
	
	while(cur_time < end_time){

		if(needs_scheduling)
		{
			scheduler->do_schedule(NUMCPUS - 1, true);
			needs_scheduling = false;
			killpg(process_group, SIGRTMIN+1);

			//wait for all tasks to actually finish rescheduling
			for (int i = 0; i < scheduler->get_schedule()->count(); i++){

				while(scheduler->get_schedule()->get_task(i)->check_mode_transition() == false){

					get_time(&cur_time);
					
					if (cur_time > end_time)
						break;

				}
		
			}
		
		}

		get_time(&cur_time);

	}

}

void force_cleanup(bool kill_processes = true) {

	if (kill_processes) {

		//kill all children
		kill(0, SIGTERM);
		
	}

	scheduler->setTermination();
	process_barrier::destroy_barrier(barrier_name);
	process_barrier::destroy_barrier(barrier_name2);

	//delete the shared memory
	smm::delete_memory<int>("PRSEDOBJ");

	if (explicit_sync) {
		process_barrier::destroy_barrier("EX_SYNC");
	}

	int queue_one, queue_two, queue_three;
	
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

	//remove those message queues
	if (msgctl(queue_one, IPC_RMID, NULL) == -1){

		print_module::print(std::cerr, "Error: Failed to remove message queue 1.\n");
		kill(0, SIGTERM);

	}

	if (msgctl(queue_two, IPC_RMID, NULL) == -1){

		print_module::print(std::cerr, "Error: Failed to remove message queue 2.\n");
		kill(0, SIGTERM);

	}

	if (msgctl(queue_three, IPC_RMID, NULL) == -1){

		print_module::print(std::cerr, "Error: Failed to remove message queue 3.\n");
		kill(0, SIGTERM);

	}
}

//System cannot be scheduled at all
void exit_unscheduable(int sig) {
	force_cleanup();
	exit(2);
}

// User requested to exit
void exit_user_request(int sig) {
	force_cleanup();
	exit(1);
}

// Child task encountered an error
void exit_from_child(int sig){
	print_module::print(std::cerr, "Signal captured from child. Schedule cannot continue. Exiting.\n");
	force_cleanup();
	exit(-1);
}

void sigrt0_handler( int signum ){
	needs_scheduling = true;
}
void sigrt1_handler( int signum ){
}

void sigrt2_handler(int signum){
	kill(0, SIGRTMIN+0);	
}          

//We use this to ensure that the clustering launcher 
//can see the signals from the tasks requesting a reschedule
void init_signal_handlers(){

	if( (signal(SIGRTMIN+0, sigrt0_handler)) == SIG_ERR ){
		print_module::print(std::cerr, "ERROR: Call to Signal failed, reason: " , strerror(errno) , "\n");
		exit(-1);
	}

	if( (signal(SIGRTMIN+1, sigrt1_handler)) == SIG_ERR ){
		print_module::print(std::cerr, "ERROR: Call to Signal failed, reason: " , strerror(errno) , "\n");
		exit(-1);
	}
}

int get_scheduling_file(std::string name, std::ifstream &ifs){

	std::string schedule_filename(name);

	//check if file is present
	ifs.open(schedule_filename);
	if(!ifs.good())
	{
		print_module::print(std::cerr, "ERROR: Cannot find schedule file: " , schedule_filename , "\n");
		return -1;
	}
	
	// Open the schedule (.rtps) file
	if (!ifs.is_open())
	{
		print_module::print(std::cerr, "ERROR: Cannot open schedule file.\n");
		return -1;
	}

	return 0;
}

struct parsed_task_mode_info {

	//mode type
	char mode_type[1024];

	//CPU stuff
	int work_sec = -1;
	int work_nsec = -1;
	int span_sec = -1;
	int span_nsec = -1;
	int period_sec = -1;
	int period_nsec = -1;

	//GPU stuff
	int gpu_work_sec = 0;
	int gpu_work_nsec = 0;
	int gpu_span_sec = 0;
	int gpu_span_nsec = 0;
	int gpu_period_sec = 0;
	int gpu_period_nsec = 0;

};

struct parsed_task_info {

	char C_program_name[1024];
	char C_program_args[1024];

	int elasticity = 1;
	int max_iterations = -1;
	// Default priority carried over from the original scheduler code - no particular reason for this otherwise
	int sched_priority = 7;
	std::vector<struct parsed_task_mode_info> modes;
};

struct parsed_shared_objects {

	int schedulable;
	unsigned sec_to_run = 0;
	long nsec_to_run = 0;

	int num_tasks_parsed = 0;
	parsed_task_info parsed_tasks[50];

	int num_modes_parsed[50] = {0};
	parsed_task_mode_info parsed_modes[50][16];

};


/************************************************************************************

Main Process:
		Acts as a file parser and general launching mechanism. It reads in the data
		contained in the .rtps file and launches the tasks basd on the periods found,
		the modes provided, and the elasticity that each task can provide. 

*************************************************************************************/
int main(int argc, char *argv[])
{
	process_group = getpgrp();
	std::vector<std::string> args(argv, argv+argc);
	//int schedulable;
	std::vector<parsed_task_info> parsed_tasks;
	unsigned sec_to_run=0;
	long nsec_to_run=0;
	
	std::ifstream ifs;
	std::string task_command_line, task_timing_line, line;
	std::vector<int> line_lengths;

	// Verify the number of arguments
	if (argc != 2)
	{

		if (std::string(argv[2]) == "SIM"){
			
			argv[2] = NULL;
			simulated_task_mode = true;

		}
		
		else {
		
			print_module::print(std::cerr, "ERROR: The program must receive a single argument which is the taskset/schedule filename without any extension.\n");
			return RT_GOMP_CLUSTERING_LAUNCHER_ARGUMENT_ERROR;
		
		}
	}

	//setup signal handlers
	init_signal_handlers();

	signal(SIGINT, exit_user_request);
	signal(SIGTERM, exit_user_request);
	signal(SIGUSR1, exit_unscheduable);
	signal(1, exit_from_child);
	
	//open the scheduling file
	if (!std::string(args[1]).ends_with(".yaml")) {
		print_module::print(std::cerr, "ERROR: RTPS files are no longer supported. Please migrate to a YAML file.\n");
		return RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR;
	}

	//open the shared memory for the stupid yaml parser
	auto yaml_object = smm::allocate<struct parsed_shared_objects>("PRSEDOBJ");

	//fork the process to allow for the yaml parser to run in a separate process
	auto pid = fork();
	if (pid == 0){

		std::cout << "starting yaml parser" << std::endl;

		//execv the yaml parser
		execv("./yaml_parser", argv);

		std::cout << "yaml parser failed to start" << std::endl;

	}

	else{

		//wait for process to return
		wait(NULL);

	}

	//open the schedule file
	if (get_scheduling_file(args[1], ifs) != 0)
	{
		print_module::print(std::cerr, "ERROR: Cannot open schedule file.\n");
		return RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR;
	}

	//read through it until we see "explicit_sync:"
	//then record the value after it
	std::string tmp_line;
	while (std::getline(ifs, tmp_line))
	{
		if (tmp_line.find("explicit_sync:") != std::string::npos)
		{
			//record if it is "true" or "false"
			if (tmp_line.find("true") != std::string::npos)
			{
				safe_calculation = true;
			}
			else if (tmp_line.find("false") != std::string::npos)
			{
				safe_calculation = false;
			}
			else
			{
				print_module::print(std::cerr, "ERROR: Invalid value for explicit_sync. Must be 'true' or 'false'.\n");
				return RT_GOMP_CLUSTERING_LAUNCHER_FILE_PARSE_ERROR;
			}
		}
	}

	//fetch all the data from the shared memory
	//schedulable = yaml_object->schedulable;
	sec_to_run = yaml_object->sec_to_run;
	nsec_to_run = yaml_object->nsec_to_run;

	for (int i = 0; i < yaml_object->num_tasks_parsed; i++){

		struct parsed_task_info task_info;

		task_info.elasticity = yaml_object->parsed_tasks[i].elasticity;
		task_info.max_iterations = yaml_object->parsed_tasks[i].max_iterations;
		task_info.sched_priority = yaml_object->parsed_tasks[i].sched_priority;

		for (int j = 0; j < 1024; j++){
			
			task_info.C_program_name[j] = yaml_object->parsed_tasks[i].C_program_name[j];
			task_info.C_program_args[j] = yaml_object->parsed_tasks[i].C_program_args[j];

		}

		for (int j = 0; j < yaml_object->num_modes_parsed[i]; j++){

			struct parsed_task_mode_info mode_info;

			mode_info.work_sec = yaml_object->parsed_modes[i][j].work_sec;
			mode_info.work_nsec = yaml_object->parsed_modes[i][j].work_nsec;
			mode_info.span_sec = yaml_object->parsed_modes[i][j].span_sec;
			mode_info.span_nsec = yaml_object->parsed_modes[i][j].span_nsec;
			mode_info.period_sec = yaml_object->parsed_modes[i][j].period_sec;
			mode_info.period_nsec = yaml_object->parsed_modes[i][j].period_nsec;
			mode_info.gpu_work_sec = yaml_object->parsed_modes[i][j].gpu_work_sec;
			mode_info.gpu_work_nsec = yaml_object->parsed_modes[i][j].gpu_work_nsec;
			mode_info.gpu_span_sec = yaml_object->parsed_modes[i][j].gpu_span_sec;
			mode_info.gpu_span_nsec = yaml_object->parsed_modes[i][j].gpu_span_nsec;
			mode_info.gpu_period_sec = yaml_object->parsed_modes[i][j].gpu_period_sec;
			mode_info.gpu_period_nsec = yaml_object->parsed_modes[i][j].gpu_period_nsec;

			task_info.modes.push_back(mode_info);

		}

		parsed_tasks.push_back(task_info);

	}

	//create the scheduling object
	//(retain CPU 0 for the scheduler)
	scheduler = new Scheduler(parsed_tasks.size(),(int) NUMCPUS, explicit_sync, FPTAS);

	//warn if set higher than real cpu amount
	if ((NUMCPUS > std::thread::hardware_concurrency()) && !simulated_task_mode){
	
		print_module::print(std::cerr, "\n\nMORE CPUS ARE BEING USED THAN ACTUALLY EXIST. WHILE THIS IS ALLOWED FOR TESTING PURPOSES, IT IS NOT RECOMMENDED. YOUR EXECUTION WILL BE HALTED FOR 2 SECONDS TO MAKE SURE YOU SEE THIS!!!\n\n");
	
		sleep(3);

	}

	//if explicit sync is enabled, we need to create a barrier to synchronize the tasks after creation
	if (explicit_sync)
	{
		if (process_barrier::create_process_barrier("EX_SYNC", parsed_tasks.size()) == nullptr)
		{
			print_module::print(std::cerr, "ERROR: Failed to initialize barrier.\n");
			return RT_GOMP_CLUSTERING_LAUNCHER_BARRIER_INITIALIZATION_ERROR;
		}
	}


	//Initialize two barriers to synchronize the tasks after creation
	if (process_barrier::create_process_barrier(barrier_name, parsed_tasks.size() + 1) == nullptr || 
		process_barrier::create_process_barrier(barrier_name2, parsed_tasks.size() + 1) == nullptr)
	{
		print_module::print(std::cerr, "ERROR: Failed to initialize barrier.\n");
		return RT_GOMP_CLUSTERING_LAUNCHER_BARRIER_INITIALIZATION_ERROR;
	}

	//gather all time monitoring variables, and 
	//add 5 seconds to start time so all tasks have 
	//enough time to finish their init() stages.
	get_time(&current_time);
	run_time={sec_to_run, nsec_to_run};
	start_time.tv_sec = current_time.tv_sec + 8;
	start_time.tv_nsec = current_time.tv_nsec;
	end_time = start_time + run_time;

	print_module::print(std::cerr, "Explicit Sync: ", explicit_sync, "\n");

	// Iterate over the tasks, gather their arguments, timing parameters, and name 
	// and then fork and execv each one
	for (unsigned t = 0; t < parsed_tasks.size(); ++t)
	{
		struct parsed_task_info task_info = parsed_tasks[t];

		/****************************************************************************
		Construct the task arguments. Note that we use std::string and not char*, as
		character pointers would go out of scope and be overwritten before the vector
		is used.
		*****************************************************************************/
		std::vector<std::string> task_manager_argvector;

		// Add the task program name to the argument vector with number of modes as well as start and finish times
		task_manager_argvector.push_back(std::string(task_info.C_program_name));

		task_manager_argvector.push_back(std::to_string(start_time.tv_sec));
		task_manager_argvector.push_back(std::to_string(start_time.tv_nsec));

		task_manager_argvector.push_back(std::to_string(task_info.max_iterations));

		if (sec_to_run == 0 && nsec_to_run == 0) {
			task_manager_argvector.push_back(std::to_string(0));
			task_manager_argvector.push_back(std::to_string(0));	
		} else {
			task_manager_argvector.push_back(std::to_string(end_time.tv_sec));
			task_manager_argvector.push_back(std::to_string(end_time.tv_nsec));	
		}

		task_manager_argvector.push_back(std::to_string(task_info.sched_priority));

		//convert the timing parameters into the desired format

		std::vector <timespec> work;
		std::vector <timespec> span;
		std::vector <timespec> period;
		std::vector <timespec> gpu_span;
		std::vector <timespec> gpu_work;
		std::vector <timespec> gpu_period;

		for (struct parsed_task_mode_info info : task_info.modes) {
			work.push_back({info.work_sec, info.work_nsec});
			span.push_back({info.span_sec, info.span_nsec});
			period.push_back({info.period_sec, info.period_nsec});
			gpu_work.push_back({info.gpu_work_sec, info.gpu_work_nsec});
			gpu_span.push_back({info.gpu_span_sec, info.gpu_span_nsec});
			gpu_period.push_back({info.gpu_period_sec, info.gpu_period_nsec});
		}

		//Insert the task data into shared memory
		TaskData * td;
    
		td = scheduler->add_task(task_info.elasticity, task_info.modes.size(), work.data(), span.data(), period.data(), gpu_work.data(), gpu_span.data(), gpu_period.data(), safe_calculation);
		task_manager_argvector.push_back(std::to_string(td->get_index()));
		
		// Add the barrier name to the argument vector
		task_manager_argvector.push_back(barrier_name);

		// Add whether or not the system has explicit sync enabled
		task_manager_argvector.push_back(std::to_string(explicit_sync));
		
		// Add the task arguments to the argument vector
		task_manager_argvector.push_back(std::string(task_info.C_program_name));

		task_manager_argvector.push_back(std::string(task_info.C_program_args));

		// Create a vector of char * arguments from the vector of string arguments
		std::vector<const char *> task_manager_argv;
		for (std::vector<std::string>::iterator i = task_manager_argvector.begin(); i != task_manager_argvector.end(); ++i)
		{
			task_manager_argv.push_back(i->c_str());
		}
		
		//Null terminate the task manager arg vector as a sentinel
		task_manager_argv.push_back(NULL);	
		print_module::print(std::cerr, "Forking and execv-ing task " , std::string(task_info.C_program_name).c_str() , "\n");
		std::cerr << std::endl;

		//if the tasks are real
		if (!simulated_task_mode){
			
			// Fork and execv the task program
			pid_t pid = fork();
			if (pid == 0)
			{
				// Const cast is necessary for type compatibility. Since the strings are
				// not shared, there is no danger in removing the const modifier.
				execv(std::string(task_info.C_program_name).c_str(), const_cast<char **>(&task_manager_argv[0]));
				
				// Error if execv returns
				std::perror("Execv-ing a new task failed.\n");
				kill(0, SIGTERM);
				return RT_GOMP_CLUSTERING_LAUNCHER_FORK_EXECV_ERROR;
			
			}
			else if (pid == -1)
			{
				std::perror("Forking a new process for task failed,\n");
				kill(0, SIGTERM);
				return RT_GOMP_CLUSTERING_LAUNCHER_FORK_EXECV_ERROR;
			}	

		}

		//add the scheduler itself to the task table at the end
		if (t == parsed_tasks.size() - 1)
		{
			td = scheduler->add_task(task_info.elasticity, task_info.modes.size(), work.data(), span.data(), period.data(), gpu_work.data(), gpu_span.data(), gpu_period.data(), false);
			print_module::print(std::cerr, "Scheduler added to task table (parameters do not matter)\n");
		}

	}

	//tell scheduler to calculate schedule for tasks
	scheduler->do_schedule(NUMCPUS - 1);

	//if we are not simulating the run
	if (!simulated_task_mode){
	
		if (process_barrier::await_and_rearm_barrier(barrier_name2.c_str()) != 0)
		{
			print_module::print(std::cerr, "ERROR: Barrier error for scheduling task \n");
			kill(0, SIGTERM);
		}

		std::thread t(scheduler_task);
		
		// Close the file
		ifs.close();
		
		print_module::print(std::cerr, "All tasks started.\n");
		
		// Wait until all child processes have terminated
		while (!(wait(NULL) == -1 && errno == ECHILD));

		t.join();

	}

	else{

		//iteration counter
		int iter[parsed_tasks.size()];

		//microsecond tracker
		unsigned long long nanoseconds_passed = 0;



		//task run trackers
		unsigned long long current_periods_counting_down[parsed_tasks.size()];

		//simple array to track whether or not tasks have transitioned yet
		bool task_has_transitioned[parsed_tasks.size()];

		//array to store who is an instigating task, and on what iteration
		int instigating_tasks[parsed_tasks.size()];

		//memset the instigating tasks to -1
		memset(instigating_tasks, -1, sizeof(instigating_tasks));

		//arrays to store each task that is being simulated
		timespec current_period[parsed_tasks.size()];
		timespec current_work[parsed_tasks.size()];
		timespec deadline[parsed_tasks.size()];

		int current_mode[parsed_tasks.size()];
		int next_mode_to_try[parsed_tasks.size()];

		int percentile[parsed_tasks.size()];

		__uint128_t current_cpu_mask[parsed_tasks.size()];
		__uint128_t current_gpu_mask[parsed_tasks.size()];

		//for testing select 5 random tasks to be the instigators
		int task_count = parsed_tasks.size();
		int instigator_count = 5;
		int instigation_time[] = {3, 5, 7, 11, 13, 3, 5, 7, 11, 13, 3, 5, 7, 11, 13};

		//set rand seed
		srand(time(NULL));

		//select the first 5 tasks to be instigators (they are already random)
		for (int i = 0; i < instigator_count; i++){

			int random_task = i;

			instigating_tasks[random_task] = instigation_time[i];
			scheduler->get_schedule()->get_task(random_task)->set_cooperative(false);

		}

		//one control bool for scheduling gating
		bool needs_scheduling = false;
		bool reschedule_in_progress = false;

		//start each of the simulated tasks
		for (size_t i = 0; i < parsed_tasks.size(); i++){

			//call the simulated task start
			simulated_task_start(i, &current_period[i], &current_work[i], &current_mode[i], &deadline[i], &percentile[i], &current_cpu_mask[i], &current_gpu_mask[i], scheduler->get_schedule());

			//for each task, fill in what mode they are currently in 
			next_mode_to_try[i] = scheduler->get_schedule()->get_task(i)->get_current_virtual_mode();

		}

		//print the instigating tasks
		std::cout << "Instigating Task array layout: " << std::endl;
		for (size_t i = 0; i < parsed_tasks.size(); i++){

			std::cout << "Task " << i << ": " << instigating_tasks[i] << std::endl;

			current_periods_counting_down[i] = get_timespec_in_ns(current_period[i]);

			iter[i] = 0;

			task_has_transitioned[i] = false;

		}

		//for a simulation time of x iterations
		//(time simulated changes based on task parameters)
		std::cout << sec_to_run << " seconds of simulation time requested" << std::endl;
		unsigned long long time_to_run = ((unsigned long long)sec_to_run * (unsigned long long )1000000000) + (unsigned long long)nsec_to_run;
		while(nanoseconds_passed < time_to_run){

			//look through the list of task period we have and find the shortest one
			unsigned long long smallest_period = -1;
			for (size_t i = 0; i < parsed_tasks.size(); i++){
				
				unsigned long long current_period = current_periods_counting_down[i];
				
				if (current_period < smallest_period)
					smallest_period = current_period;

			}

			//update nanoseconds_passed
			nanoseconds_passed += smallest_period;

			//once we have the smallest one, we update each task's time 
			//tracker to determine which tasks are supposed to run
			for (size_t i = 0; i < parsed_tasks.size(); i++)
				current_periods_counting_down[i] -= smallest_period;

			//for each task which is ready, we execute one "loop" of it's work
			for (size_t i = 0; i < parsed_tasks.size(); i++){

				if (current_periods_counting_down[i] == 0){

					//get the task
					TaskData * td = scheduler->get_schedule()->get_task(i);

					//if this task is supposed to instigate a reschedule
					if (instigating_tasks[i] != - 1){

						if ((iter[i]++ % instigating_tasks[i] == 0) && !reschedule_in_progress){

							//tell the scheduler to reschedule
							needs_scheduling = true;

							int task_current_mode = next_mode_to_try[i];

							int mode_moving_to = ((task_current_mode + 1) % td->get_original_modes_passed());

							//print what mode we are trying to switch to
							std::cout << "Task " << i << " is instigating a reschedule to mode " << mode_moving_to << " from mode " << task_current_mode << std::endl;

							//set the mode this task should be moving into 
							//(for now just cycle through all modes)
							td->set_real_current_mode(mode_moving_to, true);

							//update the mode_moving_to
							next_mode_to_try[i] = mode_moving_to;

						}

					}

					//reset the current period counter for this task
					current_periods_counting_down[i] = get_timespec_in_ns(current_period[i]);

					//check if we had any simulated task that instigated
					//a reschedule-> I am incredibly lazy and don't want to
					//reimplement any core mechanism of the scheduler, so 
					//the way rescheduling works is that we just loop over all
					//the tasks repeatedly forcing them to handoff resources 
					//until we get a false return from all of the tasks
					if (reschedule_in_progress && !task_has_transitioned[i]){

						//call the simulated reschedule (completely fine to call this multiple times)
						task_has_transitioned[i] |= simulated_reschedule(i, &current_period[i], &current_work[i], &current_mode[i], &deadline[i], &percentile[i], &current_cpu_mask[i], &current_gpu_mask[i], scheduler->get_schedule());

					}

				}

				//check if all the tasks have transitioned yet
				if (reschedule_in_progress){

					bool all_tasks_transitioned = true;

					for (size_t i = 0; i < parsed_tasks.size(); i++)
						all_tasks_transitioned &= task_has_transitioned[i];

					if (all_tasks_transitioned){

						reschedule_in_progress = false;

					}

				}

			}

			//now we switch roles and act as the scheduler. 
			//handle any reschedule requests if they exist
			if (needs_scheduling){

				timespec scheduler_start_time;
				get_time(&scheduler_start_time);

				//call the simulated scheduler
				simulated_scheduler();

				//get the time it took to run the scheduler
				timespec scheduler_end_time;
				get_time(&scheduler_end_time);

				long long time_taken = get_timespec_in_ns(scheduler_end_time) - get_timespec_in_ns(scheduler_start_time);
				
				//set reschedule flag
				reschedule_in_progress = true;

				//set the transition array 
				for (size_t i = 0; i < parsed_tasks.size(); i++)
					task_has_transitioned[i] = false;

				needs_scheduling = false;

				//we need to simulate the time from the tasks' perspective
				//for the amount of time we spent executing the scheduler
				//for each task which is ready, we execute one "loop" of it's work
				for (size_t i = 0; i < parsed_tasks.size(); i++){

					//we need to check how many times each task should have 
					//executed while we were running the scheduler
					long long scheduler_time = (long long) current_periods_counting_down[i] - time_taken;

					if (scheduler_time < 0){
						
						//update for the first simulated run
						iter[i]++;

						//now add this tasks' period to the value until it goes positive
						while (scheduler_time < 0){

							scheduler_time += get_timespec_in_ns(current_period[i]);
							iter[i]++;

						}

					}

					//set the current period counting down to the scheduler time
					current_periods_counting_down[i] = scheduler_time;

				}

			}

		}

	}

	force_cleanup(false);
	
	return 0;
}