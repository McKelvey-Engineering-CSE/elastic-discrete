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
#include <bitset>

#include "libyaml-cpp/include/yaml-cpp/yaml.h"

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

int task_amount;

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


std::string convertToRanges(const std::string& input) {
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

void modify_self(int new_mode, int task_index){

	scheduler->get_schedule()->get_task(task_index)->set_current_mode(new_mode, true);
	needs_scheduling = true;

}


void set_cooperative(bool value, int task_index){

	scheduler->get_schedule()->get_task(task_index)->set_cooperative(value);

}

void scheduler_task()
{
	//stick this thread to CPU 0.	

	//Constantly see if we need to reschedule until time is up.
	//If we need to reschedule tasks, do it and then let them know. 
	get_time(&cur_time);

	//run first schedule
	scheduler->do_schedule();

	//run the collection code for each task
	for (int task_index = 0; task_index < task_amount; task_index++){

		//print task information
		std::ostringstream task_info;
		std::string task_header = "|  Task " + std::to_string(task_index) + " (PID: " + std::to_string(getpid()) + ") |\n";
		size_t blank_size = task_header.size();

		print_module::buffered_print(task_info, "\n", std::string(blank_size - 1, '='), "\n");
		print_module::buffered_print(task_info, task_header);
		print_module::buffered_print(task_info, std::string(blank_size - 1, '='), "\n\n");

		//cpu info
		print_module::buffered_print(task_info, "CPU Metrics: \n");
		print_module::buffered_print(task_info, "	- Lowest CPU: ", scheduler->get_schedule()->get_task(task_index)->get_current_lowest_CPU(), "\n");
		print_module::buffered_print(task_info, "	- Current CPUs: ", scheduler->get_schedule()->get_task(task_index)->get_current_CPUs(), "\n");
		print_module::buffered_print(task_info, "	- Minimum CPUs: ", scheduler->get_schedule()->get_task(task_index)->get_min_CPUs(), "\n");
		print_module::buffered_print(task_info, "	- Maximum CPUs: ", scheduler->get_schedule()->get_task(task_index)->get_max_CPUs(), "\n");
		print_module::buffered_print(task_info, "	- Practical Max: ", scheduler->get_schedule()->get_task(task_index)->get_practical_max_CPUs(), "\n\n");

		//gpu info
		print_module::buffered_print(task_info, "GPU Metrics: \n");
		print_module::buffered_print(task_info, "	- Lowest GPU: ", scheduler->get_schedule()->get_task(task_index)->get_current_lowest_GPU(), "\n");
		print_module::buffered_print(task_info, "	- Current GPUs: ", scheduler->get_schedule()->get_task(task_index)->get_current_GPUs(), "\n");
		print_module::buffered_print(task_info, "	- Minimum GPUs: ", scheduler->get_schedule()->get_task(task_index)->get_min_GPUs(), "\n");
		print_module::buffered_print(task_info, "	- Maximum GPUs: ", scheduler->get_schedule()->get_task(task_index)->get_max_GPUs(), "\n");
		print_module::buffered_print(task_info, "	- Practical Max: ", scheduler->get_schedule()->get_task(task_index)->get_practical_max_GPUs(), "\n\n");

		//timing info
		print_module::buffered_print(task_info, "Timing Metrics: \n");
		print_module::buffered_print(task_info, "	- Period s: ", scheduler->get_schedule()->get_task(task_index)->get_current_period().tv_sec , " s\n");
		print_module::buffered_print(task_info, "	- Period ns: ", scheduler->get_schedule()->get_task(task_index)->get_current_period().tv_nsec , " ns\n\n");

		std::string active_cpu_string = "";
		std::string passive_cpu_string = "";

		//for CPUs that we MAY have at some point (practical max)
		for (int j = 1; j < NUMCPUS; j++){

			//Our first CPU is our permanent CPU 
			if (j == scheduler->get_schedule()->get_task(task_index)->get_current_lowest_CPU()){
			
				cpu_set_t local_cpuset;
				CPU_ZERO(&local_cpuset);
				//CPU_SET(j, &local_cpuset);
				CPU_SET(task_index + 1, &local_cpuset);

				scheduler->get_schedule()->get_task(task_index)->set_permanent_CPU(j);

			}
			if (j >= scheduler->get_schedule()->get_task(task_index)->get_current_lowest_CPU() && j < scheduler->get_schedule()->get_task(task_index)->get_current_lowest_CPU() + scheduler->get_schedule()->get_task(task_index)->get_current_CPUs()){

				active_cpu_string += std::to_string(j) + ", ";

				scheduler->get_schedule()->get_task(task_index)->push_back_cpu(j);

			}

			else {

				passive_cpu_string += std::to_string(j) + ", ";

			}
	
		}

		//print active vs passive CPUs
		print_module::buffered_print(task_info, "CPU Core Configuration: \n");
		print_module::buffered_print(task_info, "	- Active: ", convertToRanges(active_cpu_string), "\n");
		print_module::buffered_print(task_info, "	- Passive: ", convertToRanges(passive_cpu_string), "\n\n");

		//flush all task info to terminal
		print_module::flush(std::cerr, task_info);

	}

	//vector to hold the iterations completed by each task
	std::vector<int> iterations_complete(task_amount, 0);
	
	while(iterations_complete.at(0) < 1000){

		//loop over all tasks in the schedule
		//and if they have cores to gain or give,
		//do so, since this is a simulation, no deadlocks
		//can actually form
		for (int task_index = 0; task_index < task_amount; task_index++){

			//we check to see if we are just returning resources, returning and gaining, or just gaining
			int cpu_change = scheduler->get_schedule()->get_task(task_index)->get_CPUs_change();
			int gpu_change = scheduler->get_schedule()->get_task(task_index)->get_GPUs_change();

			//giving cpus
			if (cpu_change > 0){

				//give up resources immediately and mark our transition
				for (int i = 0; i < cpu_change; i++){
					
					//remove CPUs from our set until we have given up the correct number
					scheduler->get_schedule()->get_task(task_index)->pop_back_cpu();

				}

			}

			//giving gpus
			if (gpu_change > 0){

				//give up resources immediately and mark our transition
				for (int i = 0; i < gpu_change; i++){
					
					//remove GPUs from our set until we have given up the correct number
					scheduler->get_schedule()->get_task(task_index)->pop_back_gpu();

				}

			}

			//granted resources
			auto cpus = scheduler->get_schedule()->get_task(task_index)->get_cpus_granted_from_other_tasks();
			auto gpus = scheduler->get_schedule()->get_task(task_index)->get_gpus_granted_from_other_tasks();

			//gaining cpus
			if (cpus.size() != 0){

				//collect our CPUs
				std::vector<int> core_indices;

				for (size_t i = 0; i < cpus.size(); i++)
					for (size_t j = 0; j < cpus.at(i).second.size(); j++)
						core_indices.push_back(cpus.at(i).second.at(j));
				
				//wake up the corresponding cores
				for (size_t i = 0; i < core_indices.size(); i++){

					//add them to our set
					scheduler->get_schedule()->get_task(task_index)->push_back_cpu(core_indices.at(i));

				}
				
			}

			//gaining gpus
			if (gpus.size() != 0){

				//collect our GPUs
				std::vector<int> tpc_indices;

				for (size_t i = 0; i < gpus.size(); i++)
					for (size_t j = 0; j < gpus.at(i).second.size(); j++)
						tpc_indices.push_back(gpus.at(i).second.at(j));
				
				//wake up the corresponding SMs
				for (size_t i = 0; i < tpc_indices.size(); i++){

					//add them to our set
					scheduler->get_schedule()->get_task(task_index)->push_back_gpu(tpc_indices.at(i));

				}

			}

			//clear our allocation amounts
			scheduler->get_schedule()->get_task(task_index)->clear_cpus_granted_from_other_tasks();
			scheduler->get_schedule()->get_task(task_index)->clear_gpus_granted_from_other_tasks();

			//clear our change amount as well
			scheduler->get_schedule()->get_task(task_index)->set_CPUs_change(0);
			scheduler->get_schedule()->get_task(task_index)->set_GPUs_change(0);

			//update our cpu mask
			auto current_cpu_mask = scheduler->get_schedule()->get_task(task_index)->get_cpu_mask();

			print_module::task_print(std::cerr, (unsigned long long) current_cpu_mask, "\n");

		}

		//for each task present, simulate their execution
		for (int task_index = 0; task_index < scheduler->get_num_tasks(); task_index++){

			if (task_index < 3)
				set_cooperative(false, task_index);

			std::vector<int> intervals = {3, 5, 7};

			std::bitset<128> thread_mask(scheduler->get_schedule()->get_task(task_index)->get_cpu_mask());

			int count = 0;

			// Wake up the correct threads
			for (size_t i = 1; i < 128; ++i)
				if (thread_mask[i])
					count ++;

			std::cout << "TEST: [" << task_index << "," << iterations_complete.at(task_index) << "] core count: " << count << std::endl;

			iterations_complete.at(task_index)++;

			if (task_index < 3)
					if (iterations_complete.at(task_index) % intervals.at(task_index) == 0 && iterations_complete.at(task_index) % 2 == 1)
						modify_self(1, task_index);

			if (task_index < 3)
					if (iterations_complete.at(task_index) % intervals.at(task_index) == 0 && iterations_complete.at(task_index) % 2 == 0)
						modify_self(3, task_index);
		}

		if(needs_scheduling)
		{

			scheduler->do_schedule();
			needs_scheduling = false;
		
		}

		get_time(&cur_time);

	}

}

void force_cleanup() {
	scheduler->setTermination();
	process_barrier::destroy_barrier(barrier_name);
	process_barrier::destroy_barrier(barrier_name2);

	//if (explicit_sync) {
		process_barrier::destroy_barrier("EX_SYNC");
	//}

	kill(0, SIGKILL);
	delete scheduler;
}

// User requested to exit
void exit_user_request(int sig) {
	force_cleanup();
	exit(0);
}

// Child task encountered an error
void exit_from_child(int sig){
	print_module::print(std::cerr, "Signal captured from child. Schedule cannot continue. Exiting.\n");
	force_cleanup();
	exit(0);
}

void sigrt0_handler( int signum ){
	needs_scheduling = true;
}
void sigrt1_handler( int signum ){
}

void sigrt2_handler(int signum){
	kill(0, SIGRTMIN+0);	
}          

void init_signal_handlers(){
	//Set up a signal handler for SIGRT0
	void (*ret_handler)(int);

	if( (ret_handler = signal(SIGRTMIN+0, sigrt0_handler)) == SIG_ERR ){
		print_module::print(std::cerr, "ERROR: Call to Signal failed, reason: " , strerror(errno) , "\n");
		exit(-1);
	}

	if( (ret_handler = signal(SIGRTMIN+1, sigrt1_handler)) == SIG_ERR ){
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
	std::string mode_type = "";

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
	std::string program_name = "";
	std::string program_args = "";
	int elasticity = 1;
	int max_iterations = -1;
	// Default priority carried over from the original scheduler code - no particular reason for this otherwise
	int sched_priority = 7;
	std::vector<struct parsed_task_mode_info> modes;
};

bool yaml_is_time(YAML::Node node) {
	return node && node["sec"] && node["nsec"];
}

int read_scheduling_yaml_file(std::ifstream &ifs, 
						int* schedulable, 
						std::vector<parsed_task_info>* parsed_tasks, 
						unsigned* sec_to_run, 
						long* nsec_to_run){
	std::stringstream buffer;
	buffer << ifs.rdbuf();
	std::string contents = buffer.str();

	YAML::Node config = YAML::Load(contents);
	
	if (!config["schedulable"]) {
		return -1;
	}
	if (config["explicit_sync"]){
		explicit_sync = config["explicit_sync"].as<bool>();
	}
	if (config["FPTAS"]){
		FPTAS = config["FPTAS"].as<bool>();
	}
	if (config["schedulable"].as<bool>()) {
		*schedulable = 1;
	} else {
		*schedulable = 0;
	}

	if (yaml_is_time(config["maxRuntime"])) {
		*sec_to_run = config["maxRuntime"]["sec"].as<unsigned int>();
		*nsec_to_run = config["maxRuntime"]["nsec"].as<unsigned int>();
	} else {
		*sec_to_run = 0;
		*nsec_to_run = 0;
	}

	if (!config["tasks"]) {
		return -3;
	}
	for (YAML::Node task : config["tasks"]) {
		struct parsed_task_info task_info;

		if (!task["program"] || !task["program"]["name"]) {
			return -4;
		}
		
		task_info.program_name = task["program"]["name"].as<std::string>();
		if (task["program"]["args"]) {
			task_info.program_args = task["program"]["args"].as<std::string>();
		} else {
			task_info.program_args = "";
		}


		if (task["elasticity"]) {
			task_info.elasticity = task["elasticity"].as<int>();
		}

		if (task["priority"]) {
			task_info.sched_priority = task["priority"].as<int>();
		}


		if (task["maxIterations"]) {
			task_info.max_iterations = task["maxIterations"].as<int>();
		} else if (*sec_to_run == 0) {
			print_module::print(std::cout, "Warning: no maxRuntime set and task has no maxIterations; will run forever\n");
		}

		if (!task["modes"] || !task["modes"].IsSequence() || task["modes"].size() == 0) {
			return -5;
		}

		for (YAML::Node mode : task["modes"]) {
			if (!yaml_is_time(mode["work"]) || !yaml_is_time(mode["span"]) || !yaml_is_time(mode["period"])) {
				return -6;
			}

			struct parsed_task_mode_info mode_info;

			if (mode["type"])
				mode_info.mode_type = mode["type"].as<std::string>();
			else
				mode_info.mode_type = "heavy";

			mode_info.work_sec = mode["work"]["sec"].as<int>();
			mode_info.work_nsec = mode["work"]["nsec"].as<int>();
			mode_info.span_sec = mode["span"]["sec"].as<int>();
			mode_info.span_nsec = mode["span"]["nsec"].as<int>();
			mode_info.period_sec = mode["period"]["sec"].as<int>();
			mode_info.period_nsec = mode["period"]["nsec"].as<int>();

			//check for GPU params
			if (yaml_is_time(mode["gpu_work"]) && yaml_is_time(mode["gpu_span"])){

				mode_info.gpu_work_sec = mode["gpu_work"]["sec"].as<int>();
				mode_info.gpu_work_nsec = mode["gpu_work"]["nsec"].as<int>();
				mode_info.gpu_span_sec = mode["gpu_span"]["sec"].as<int>();
				mode_info.gpu_span_nsec = mode["gpu_span"]["nsec"].as<int>();

				//if an arbitrary period has been set for modifying the GPU assignemnt
				if (yaml_is_time(mode["gpu_period"])){

					mode_info.gpu_period_sec = mode["gpu_period"]["sec"].as<int>();
					mode_info.gpu_period_nsec = mode["gpu_period"]["nsec"].as<int>();

				}

				else{

					mode_info.gpu_period_sec = mode_info.period_sec;
					mode_info.gpu_period_nsec = mode_info.period_nsec;
				}

			}

			task_info.modes.push_back(mode_info);
		}


		parsed_tasks->push_back(task_info);
	}

	return 0;
}

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
	int schedulable;
	std::vector<parsed_task_info> parsed_tasks;
	unsigned sec_to_run=0;
	long nsec_to_run=0;
	
	std::ifstream ifs;
	std::string task_command_line, task_timing_line, line;
	std::vector<int> line_lengths;

	// Verify the number of arguments
	if (argc != 2)
	{
		print_module::print(std::cerr, "ERROR: The program must receive a single argument which is the taskset/schedule filename without any extension.\n");
		return RT_GOMP_CLUSTERING_LAUNCHER_ARGUMENT_ERROR;
	}

	//setup signal handlers
	init_signal_handlers();

	signal(SIGINT, exit_user_request);
	signal(SIGTERM, exit_user_request);
	signal(1, exit_from_child);
	
	//open the scheduling file

	if (!std::string(args[1]).ends_with(".yaml")) {
		print_module::print(std::cerr, "ERROR: RTPS files are no longer supported. Please migrate to a YAML file.\n");
		return RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR;
	}

	if (get_scheduling_file(args[1], ifs) != 0) return RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR;

	if (read_scheduling_yaml_file(ifs, &schedulable, &parsed_tasks, &sec_to_run, &nsec_to_run) != 0) return RT_GOMP_CLUSTERING_LAUNCHER_FILE_PARSE_ERROR;

	//create the scheduling object
	//(retain CPU 0 for the scheduler)
	scheduler = new Scheduler(parsed_tasks.size(),(int) NUMCPUS, explicit_sync, FPTAS);

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
		task_manager_argvector.push_back(task_info.program_name);
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
	
		//FIXME: GPU INFO JUST USES THE CPU PORTION OF THE INFO. REPLACE WITH REAL INFORMATION
		td = scheduler->add_task(task_info.elasticity, task_info.modes.size(), work.data(), span.data(), period.data(), gpu_work.data(), gpu_span.data(), gpu_period.data());
		task_manager_argvector.push_back(std::to_string(td->get_index()));
		
		// Add the barrier name to the argument vector
		task_manager_argvector.push_back(barrier_name);

		// Add whether or not the system has explicit sync enabled
		task_manager_argvector.push_back(std::to_string(explicit_sync));
		
		// Add the task arguments to the argument vector
		task_manager_argvector.push_back(task_info.program_name);

		task_manager_argvector.push_back(task_info.program_args);

		// Create a vector of char * arguments from the vector of string arguments
		std::vector<const char *> task_manager_argv;
		for (std::vector<std::string>::iterator i = task_manager_argvector.begin(); i != task_manager_argvector.end(); ++i)
		{
			task_manager_argv.push_back(i->c_str());
		}
		
	}

	task_amount = parsed_tasks.size();

	std::thread t(scheduler_task);
	
	// Close the file
	ifs.close();
	
	print_module::print(std::cerr, "All tasks started.\n");
	
	// Wait until all child processes have terminated
	while (!(wait(NULL) == -1 && errno == ECHILD));

	t.join();

	print_module::print(std::cerr, "All tasks finished.\n");

	delete scheduler;
	
	return 0;
}

