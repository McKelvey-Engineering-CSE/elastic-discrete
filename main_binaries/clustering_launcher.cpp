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
			scheduler->do_schedule();
			needs_scheduling = false;
			killpg(process_group, SIGRTMIN+1);

			//wait for all tasks to actually finish rescheduling
			for (int i = 0; i < scheduler->get_num_tasks(); i++){

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

void force_cleanup() {
	scheduler->setTermination();
	process_barrier::destroy_barrier(barrier_name);
	process_barrier::destroy_barrier(barrier_name2);

	if (explicit_sync) {
		process_barrier::destroy_barrier("EX_SYNC");
	}

	kill(0, SIGKILL);
	//delete scheduler;
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

	//warn if set higher than real cpu amount
	if (NUMCPUS > std::thread::hardware_concurrency()){
	
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
		
		//Null terminate the task manager arg vector as a sentinel
		task_manager_argv.push_back(NULL);	
		print_module::print(std::cerr, "Forking and execv-ing task " , task_info.program_name.c_str() , "\n");
		
		// Fork and execv the task program
		pid_t pid = fork();
		if (pid == 0)
		{
			// Const cast is necessary for type compatibility. Since the strings are
			// not shared, there is no danger in removing the const modifier.
			execv(task_info.program_name.c_str(), const_cast<char **>(&task_manager_argv[0]));
			
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

	//run the table generation for all unsafe task combinations
	scheduler->generate_unsafe_combinations();

	//tell scheduler to calculate schedule for tasks
	scheduler->do_schedule();
	
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

	print_module::print(std::cerr, "All tasks finished.\n");

	if (explicit_sync) {
		process_barrier::destroy_barrier("EX_SYNC");
	}
	
	scheduler->setTermination();
	process_barrier::destroy_barrier(barrier_name);
	process_barrier::destroy_barrier(barrier_name2);
	
	return 0;
}

