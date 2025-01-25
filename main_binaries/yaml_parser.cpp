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

bool needs_scheduling = false;
Scheduler * scheduler;
pid_t process_group;

//BOOLEAN VALUE FOR WHETHER WE INITIALLY INCLUDE LAST TASK IN THE SCHEDULE
bool add_last_task = true;
TaskData * last_task;

//bool which controls whether or not we are running with explicit syncronization
bool explicit_sync = false;
bool FPTAS = false;

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
		
		std::string temp_name = task["program"]["name"].as<std::string>();

		// Copy the program name and args to C strings for execv
		strcpy(task_info.C_program_name, temp_name.c_str());

		if (task["program"]["args"]) {
			temp_name = task["program"]["args"].as<std::string>();
			strcpy(task_info.C_program_args, temp_name.c_str());
		} else {
			temp_name = "";
			task_info.C_program_args[0] = '\0';
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

			if (mode["type"]){
				
				std::string mode_temp = mode["type"].as<std::string>();
				strcpy(mode_info.mode_type, mode_temp.c_str());

			}

			else
				strcpy(mode_info.mode_type, "heavy");

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
	
	//open the scheduling file
	if (!std::string(args[1]).ends_with(".yaml")) {
		print_module::print(std::cerr, "ERROR: RTPS files are no longer supported. Please migrate to a YAML file.\n");
		return RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR;
	}

	//open the shared memory for the stupid yaml parser
	auto yaml_object = smm::fetch<struct parsed_shared_objects>("PRSEDOBJ");

	if (get_scheduling_file(args[1], ifs) != 0) return RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR;

	if (read_scheduling_yaml_file(ifs, &schedulable, &parsed_tasks, &sec_to_run, &nsec_to_run) != 0) return RT_GOMP_CLUSTERING_LAUNCHER_FILE_PARSE_ERROR;

	//move all the data to the shared memory
	yaml_object->schedulable = schedulable;
	yaml_object->sec_to_run = sec_to_run;
	yaml_object->nsec_to_run = nsec_to_run;
	yaml_object->num_tasks_parsed = parsed_tasks.size();

	for (int i = 0; i < parsed_tasks.size(); i++) {

		yaml_object->parsed_tasks[i] = parsed_tasks[i];

		yaml_object->num_modes_parsed[i] = parsed_tasks[i].modes.size();

		for (int j = 0; j < parsed_tasks[i].modes.size(); j++) {

			yaml_object->parsed_modes[i][j] = parsed_tasks[i].modes[j];

		}

	}

	std::cout << "YAML PARSER: Finished parsing the YAML file." << std::endl;

	return 0;
}

