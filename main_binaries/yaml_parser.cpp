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
#include <cstring>

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

	//CPU C stuff
	int cpu_C_work_sec = 0;
	int cpu_C_work_nsec = 0;
	int cpu_C_span_sec = 0;
	int cpu_C_span_nsec = 0;
	int cpu_C_period_sec = 0;
	int cpu_C_period_nsec = 0;

	//GPU D stuff
	int gpu_D_work_sec = 0;
	int gpu_D_work_nsec = 0;
	int gpu_D_span_sec = 0;
	int gpu_D_span_nsec = 0;
	int gpu_D_period_sec = 0;
	int gpu_D_period_nsec = 0;

};

struct parsed_task_info {

	char C_program_name[1024];
	char C_program_args[1024];

	int elasticity = 1;
	int max_iterations = -1;
	// Default priority carried over from the original scheduler code - no particular reason for this otherwise
	int sched_priority = 7;
	
	// Per-task processor equivalence configuration
	// processor_type: 0=A, 1=B, 2=C, 3=D
	// ratio: scaling factor relative to the equivalent processor
	int processor_A_type = 0;
	double processor_A_ratio = 1.0;
	int processor_B_type = 1;
	double processor_B_ratio = 1.0;
	int processor_C_type = 2;
	double processor_C_ratio = 1.0;
	int processor_D_type = 3;
	double processor_D_ratio = 1.0;
	
	std::vector<struct parsed_task_mode_info> modes;
};

struct parsed_shared_objects {

	int schedulable;
	unsigned sec_to_run = 0;
	long nsec_to_run = 0;

	// Processor topology configuration
	// Maps each processor type to its physical topology
	int processor_topology_A = 0;  // 0=A, 1=B, 2=C, 3=D
	int processor_topology_B = 1;
	int processor_topology_C = 2;
	int processor_topology_D = 3;

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
						long* nsec_to_run,
						int* proc_topology_A, int* proc_topology_B, 
						int* proc_topology_C, int* proc_topology_D){
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

	// Parse processor topology configuration
	if (config["processor_topology"]) {
		YAML::Node topology_config = config["processor_topology"];
		
		// Parse each processor type (A, B, C, D)
		for (YAML::const_iterator it = topology_config.begin(); it != topology_config.end(); ++it) {
			std::string proc_type = it->first.as<std::string>();
			std::string topology_value = it->second.as<std::string>();
			
			// Map topology values to indices (0=A, 1=B, 2=C, 3=D)
			int topology_index = -1;
			if (topology_value == "A") topology_index = 0;
			else if (topology_value == "B") topology_index = 1;
			else if (topology_value == "C") topology_index = 2;
			else if (topology_value == "D") topology_index = 3;
			else {
				std::cerr << "Invalid topology value: " << topology_value << std::endl;
				continue;
			}
			
			// Store the topology configuration based on processor type
			if (proc_type == "A") {
				*proc_topology_A = topology_index;
			} else if (proc_type == "B") {
				*proc_topology_B = topology_index;
			} else if (proc_type == "C") {
				*proc_topology_C = topology_index;
			} else if (proc_type == "D") {
				*proc_topology_D = topology_index;
			} else {
				std::cerr << "Invalid processor type: " << proc_type << std::endl;
				continue;
			}
			
			// Debug output
			std::cout << "Parsed processor topology: " << proc_type << " -> " << topology_value 
				<< " (index=" << topology_index << ")" << std::endl;
		}
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

		// Parse per-task processor equivalence configuration
		if (task["processor_equivalence"]) {
			YAML::Node proc_equiv = task["processor_equivalence"];
			
			// Parse each processor type (A, B, C, D)
			for (YAML::const_iterator it = proc_equiv.begin(); it != proc_equiv.end(); ++it) {
				std::string proc_type = it->first.as<std::string>();
				std::string proc_value = it->second.as<std::string>();
				
				// Extract the equivalent processor type and ratio
				// Format is like "0.8A" or "1A" or "1C"
				char equivalent_proc = proc_value.back(); // Get the last character (A, B, C, or D)
				std::string ratio_str = proc_value.substr(0, proc_value.length() - 1);
				double ratio = 1.0; // Default ratio
				if (!ratio_str.empty()) {
					try {
						ratio = std::stod(ratio_str);
					} catch (const std::exception& e) {
						std::cerr << "Error parsing ratio from '" << ratio_str << "' for processor " << proc_type << std::endl;
						ratio = 1.0;
					}
				}
				
				// Map processor types to indices (0=A, 1=B, 2=C, 3=D)
				int proc_index = -1;
				int equivalent_index = -1;
				
				if (proc_type == "A") proc_index = 0;
				else if (proc_type == "B") proc_index = 1;
				else if (proc_type == "C") proc_index = 2;
				else if (proc_type == "D") proc_index = 3;
				else {
					std::cerr << "Invalid processor type: " << proc_type << std::endl;
					continue;
				}
				
				if (equivalent_proc == 'A') equivalent_index = 0;
				else if (equivalent_proc == 'B') equivalent_index = 1;
				else if (equivalent_proc == 'C') equivalent_index = 2;
				else if (equivalent_proc == 'D') equivalent_index = 3;
				else {
					std::cerr << "Invalid equivalent processor: " << equivalent_proc << std::endl;
					continue;
				}
				
				// Store the configuration based on processor type
				if (proc_index == 0) { // A
					task_info.processor_A_type = equivalent_index;
					task_info.processor_A_ratio = ratio;
				} else if (proc_index == 1) { // B
					task_info.processor_B_type = equivalent_index;
					task_info.processor_B_ratio = ratio;
				} else if (proc_index == 2) { // C
					task_info.processor_C_type = equivalent_index;
					task_info.processor_C_ratio = ratio;
				} else if (proc_index == 3) { // D
					task_info.processor_D_type = equivalent_index;
					task_info.processor_D_ratio = ratio;
				}
				
				// Debug output
				std::cout << "Parsed task processor config: " << proc_type << " -> " << proc_value 
					<< " (type=" << equivalent_index << ", ratio=" << ratio << ")" << std::endl;
			}
		}

		if (task["maxIterations"]) {
			task_info.max_iterations = task["maxIterations"].as<int>();
		} else if (*sec_to_run == 0) {
			print_module::print(std::cout, "Warning: no maxRuntime set and task has no maxIterations; will run forever\n");
		}

		bool has_modes = true;
		bool has_ranges = true;

		if (!task["modes"] || !task["modes"].IsSequence() || task["modes"].size() == 0) {
			has_modes = false;
		}

		if (!task["ranges"] || !task["ranges"].IsSequence() || task["ranges"].size() == 0) {
			has_ranges = false;
		}

		if (!has_modes && !has_ranges) {
			return -5;
		}

		if (has_modes){

			for (YAML::Node mode : task["modes"]) {
				if (!yaml_is_time(mode["work_A"]) || !yaml_is_time(mode["span_A"]) || !yaml_is_time(mode["period"])) {
					return -6;
				}

				struct parsed_task_mode_info mode_info;

				if (mode["type"]){
					
					std::string mode_temp = mode["type"].as<std::string>();
					strcpy(mode_info.mode_type, mode_temp.c_str());

				}

				else
					strcpy(mode_info.mode_type, "heavy");

				mode_info.work_sec = mode["work_A"]["sec"].as<int>();
				mode_info.work_nsec = mode["work_A"]["nsec"].as<int>();
				mode_info.span_sec = mode["span_A"]["sec"].as<int>();
				mode_info.span_nsec = mode["span_A"]["nsec"].as<int>();
				mode_info.period_sec = mode["period"]["sec"].as<int>();
				mode_info.period_nsec = mode["period"]["nsec"].as<int>();

				//check for GPU params
				if (yaml_is_time(mode["work_B"]) && yaml_is_time(mode["span_B"])){

					mode_info.gpu_work_sec = mode["work_B"]["sec"].as<int>();
					mode_info.gpu_work_nsec = mode["work_B"]["nsec"].as<int>();
					mode_info.gpu_span_sec = mode["span_B"]["sec"].as<int>();
					mode_info.gpu_span_nsec = mode["span_B"]["nsec"].as<int>();

				}

				//check for CPU C params
				if (yaml_is_time(mode["work_C"]) && yaml_is_time(mode["span_C"])){

					mode_info.cpu_C_work_sec = mode["work_C"]["sec"].as<int>();
					mode_info.cpu_C_work_nsec = mode["work_C"]["nsec"].as<int>();
					mode_info.cpu_C_span_sec = mode["span_C"]["sec"].as<int>();
					mode_info.cpu_C_span_nsec = mode["span_C"]["nsec"].as<int>();

				}

				//check for GPU D params
				if (yaml_is_time(mode["work_D"]) && yaml_is_time(mode["span_D"])){

					mode_info.gpu_D_work_sec = mode["work_D"]["sec"].as<int>();
					mode_info.gpu_D_work_nsec = mode["work_D"]["nsec"].as<int>();
					mode_info.gpu_D_span_sec = mode["span_D"]["sec"].as<int>();
					mode_info.gpu_D_span_nsec = mode["span_D"]["nsec"].as<int>();

				}

				task_info.modes.push_back(mode_info);
			}

		}

		else if (has_ranges){

			std::vector<struct parsed_task_mode_info> elastic_ranges;

			for (YAML::Node mode : task["ranges"]) {
				if (!yaml_is_time(mode["work_A"]) || !yaml_is_time(mode["span_A"]) || !yaml_is_time(mode["period"])) {
					return -6;
				}

				struct parsed_task_mode_info mode_info;

				if (mode["type"]){
					
					std::string mode_temp = mode["type"].as<std::string>();
					strcpy(mode_info.mode_type, mode_temp.c_str());

				}

				else
					strcpy(mode_info.mode_type, "heavy");

				mode_info.work_sec = mode["work_A"]["sec"].as<int>();
				mode_info.work_nsec = mode["work_A"]["nsec"].as<int>();
				mode_info.span_sec = mode["span_A"]["sec"].as<int>();
				mode_info.span_nsec = mode["span_A"]["nsec"].as<int>();
				mode_info.period_sec = mode["period"]["sec"].as<int>();
				mode_info.period_nsec = mode["period"]["nsec"].as<int>();

				//check for GPU params
				if (yaml_is_time(mode["work_B"]) && yaml_is_time(mode["span_B"])){

					mode_info.gpu_work_sec = mode["work_B"]["sec"].as<int>();
					mode_info.gpu_work_nsec = mode["work_B"]["nsec"].as<int>();
					mode_info.gpu_span_sec = mode["span_B"]["sec"].as<int>();
					mode_info.gpu_span_nsec = mode["span_B"]["nsec"].as<int>();

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

				//check for CPU C params
				if (yaml_is_time(mode["work_C"]) && yaml_is_time(mode["span_C"])){

					mode_info.cpu_C_work_sec = mode["work_C"]["sec"].as<int>();
					mode_info.cpu_C_work_nsec = mode["work_C"]["nsec"].as<int>();
					mode_info.cpu_C_span_sec = mode["span_C"]["sec"].as<int>();
					mode_info.cpu_C_span_nsec = mode["span_C"]["nsec"].as<int>();

					//if an arbitrary period has been set for modifying the CPU C assignment
					if (yaml_is_time(mode["cpu_C_period"])){

						mode_info.cpu_C_period_sec = mode["cpu_C_period"]["sec"].as<int>();
						mode_info.cpu_C_period_nsec = mode["cpu_C_period"]["nsec"].as<int>();

					}

					else{

						mode_info.cpu_C_period_sec = mode_info.period_sec;
						mode_info.cpu_C_period_nsec = mode_info.period_nsec;
					}

				}

				//check for GPU D params
				if (yaml_is_time(mode["work_D"]) && yaml_is_time(mode["span_D"])){

					mode_info.gpu_D_work_sec = mode["work_D"]["sec"].as<int>();
					mode_info.gpu_D_work_nsec = mode["work_D"]["nsec"].as<int>();
					mode_info.gpu_D_span_sec = mode["span_D"]["sec"].as<int>();
					mode_info.gpu_D_span_nsec = mode["span_D"]["nsec"].as<int>();

					//if an arbitrary period has been set for modifying the GPU D assignment
					if (yaml_is_time(mode["gpu_D_period"])){

						mode_info.gpu_D_period_sec = mode["gpu_D_period"]["sec"].as<int>();
						mode_info.gpu_D_period_nsec = mode["gpu_D_period"]["nsec"].as<int>();

					}

					else{

						mode_info.gpu_D_period_sec = mode_info.period_sec;
						mode_info.gpu_D_period_nsec = mode_info.period_nsec;
					}

				}

				elastic_ranges.push_back(mode_info);

			}

			//regardless of how many ranges are provided, we only take the first two
			//with the assumption that the first is the lightest and the second is the heaviest
			std::vector<timespec> period;
			std::vector<timespec> work_A;
			std::vector<timespec> span_A;
			std::vector<timespec> GPU_work;
			std::vector<timespec> GPU_span;

			timespec numerator;
			timespec denominator;

			for (int i = 0; i < 2; i++){

				period.push_back({elastic_ranges[i].period_sec, elastic_ranges[i].period_nsec});
				work_A.push_back({elastic_ranges[i].work_sec, elastic_ranges[i].work_nsec});
				span_A.push_back({elastic_ranges[i].span_sec, elastic_ranges[i].span_nsec});
				GPU_work.push_back({elastic_ranges[i].gpu_work_sec, elastic_ranges[i].gpu_work_nsec});
				GPU_span.push_back({elastic_ranges[i].gpu_span_sec, elastic_ranges[i].gpu_span_nsec});

			}

			//Determine what is actually being adjusted here:
			//the work_A or the period
			bool period_elastic = false;
			bool work_elastic = false;
			bool is_pure_cpu_task = false;

			//check if it's a purely cpu taks
			if (GPU_work[0] == timespec({0, 0}) &&  GPU_span[0] == timespec({0, 0}) && GPU_work[1] == timespec({0, 0}) &&  GPU_span[1] == timespec({0, 0}))
				is_pure_cpu_task = true;

			//determine if the period is being adjusted
			if (period[0] != period[1])
				period_elastic = true;

			//determine if the work_A is being adjusted
			if (work_A[0] != work_A[1])
				work_elastic = true;

			if (period_elastic && work_elastic){

				print_module::print(std::cerr, "ERROR: Both period and work_A are being adjusted on a continuous elastic task. This is not supported.\n");
				return -7;

			}

			else if (!period_elastic && !work_elastic){

				print_module::print(std::cerr, "ERROR: Neither period nor work_A are being adjusted on a continuous elastic task. This is not supported.\n");
				return -8;

			}

			//NOTE FOR LATER: this is not how period elasticity works.
			//because the task has a cpu and gpu to work_A with in terms of work_A and span_A
			//but shares a period, there are certain configurations which are locked in
			//terms of what resources the task can use. For pure cpu tasks, what is shown
			//below will work_A, and for all work_A elastic tasks it should work_A as well,
			//but for period elastic tasks, we need to walk the period from min to max
			//and find mappings for all possible configurations of cpus we can end up with.
			else if (period_elastic){

				//we want to walk the period from min to max and find mappings for all 
				//possible configurations of cpus we can end up with.
				if (is_pure_cpu_task){

					//calc the minimum and maximum cpu allocations
					int min_CPUs = 0;
					int max_CPUs = 0;

					for (int i = 0; i < 2; i++){

						ts_diff(work_A[i], span_A[i], numerator);
						ts_diff(period[i], span_A[i], denominator);

						int CPUs = (int)ceil(numerator / denominator);

						if (CPUs > max_CPUs)
							max_CPUs = CPUs;

						if (CPUs < min_CPUs)
							min_CPUs = CPUs;

					}

					//set numerator again just so it's easier to see
					//when looking back at this code
					ts_diff(work_A[0], span_A[0], numerator);

					//now we just solve for the periods which 
					//can make the cpu allocations which fall
					//between min and max
					for (int i = min_CPUs; i <= max_CPUs; i++){

						//determine the period
						//by solving for the period
						//which can give us the desired
						//CPU allocation
						long long numerator_as_value = get_timespec_in_ns(numerator);
						numerator_as_value /= i;

						//get the span_A as a value
						long long span_as_value = get_timespec_in_ns(span_A[0]);

						long long target_period = numerator_as_value + span_as_value;

						//make a mode out of this
						struct parsed_task_mode_info mode_info;

						strcpy(mode_info.mode_type, "heavy");

						mode_info.work_sec = work_A[0].tv_sec;
						mode_info.work_nsec = work_A[0].tv_nsec;
						mode_info.span_sec = span_A[0].tv_sec;
						mode_info.span_nsec = span_A[0].tv_nsec;
						mode_info.period_sec = target_period / 1000000000;
						mode_info.period_nsec = target_period % 1000000000;

						//push the mode
						task_info.modes.push_back(mode_info);

					}

				}

				//if not a pure cpu task, then we do the same thing for both 
				//the cpu and the gpu resources with the modified period
				else{

					std::vector<long> nanosecond_periods;

					//calc the minimum and maximum processor allocations
					int min_CPUs = 100000;
					int max_CPUs = 0;

					int min_GPUs = 100000;
					int max_GPUs = 0;

					for (int i = 0; i < 2; i++){

						auto modified_period = period[i] * 0.5;

						ts_diff(work_A[i], span_A[i], numerator);
						ts_diff(modified_period, span_A[i], denominator);

						int CPUs = (int)ceil(numerator / denominator);

						if (CPUs > max_CPUs)
							max_CPUs = CPUs;

						if (CPUs < min_CPUs)
							min_CPUs = CPUs;

						ts_diff(GPU_work[i], GPU_span[i], numerator);
						ts_diff(modified_period, GPU_span[i], denominator);

						int GPUs = (int)ceil(numerator / denominator);

						if (GPUs > max_GPUs)
							max_GPUs = GPUs;

						if (GPUs < min_GPUs)
							min_GPUs = GPUs;

					}

					//set numerator again just so it's easier to see
					//when looking back at this code
					ts_diff(work_A[0], span_A[0], numerator);

					//now we just solve for the periods which 
					//can make the cpu allocations which fall
					//between min and max
					for (int i = min_CPUs; i <= max_CPUs; i++){

						//determine the period
						//by solving for the period
						//which can give us the desired
						//CPU allocation
						long long numerator_as_value = get_timespec_in_ns(numerator);
						numerator_as_value /= i;

						//get the span_A as a value
						long long span_as_value = get_timespec_in_ns(span_A[0]);

						long long target_period = numerator_as_value + span_as_value;

						//add it to the candidate periods
						nanosecond_periods.push_back(target_period);

					}

					//do the same procedure but now to get all the possible GPU allocations
					ts_diff(GPU_work[0], GPU_span[0], numerator);

					for (int i = min_GPUs; i <= max_GPUs; i++){

						//determine the period
						//by solving for the period
						//which can give us the desired
						//CPU allocation
						long long numerator_as_value = get_timespec_in_ns(numerator);
						numerator_as_value /= i;

						//get the span_A as a value
						long long span_as_value = get_timespec_in_ns(GPU_span[0]);

						long long target_period = numerator_as_value + span_as_value;

						//add it to the candidate periods
						nanosecond_periods.push_back(target_period);

					}

					//now loop over all the candidate values and detemrine which ones 
					//are unique. Any that are not we do not keep
					std::vector<std::pair<int, int>> unique_modes;
					for (size_t i = 0; i < nanosecond_periods.size(); i++){

						auto modified_period = timespec_from_ns(nanosecond_periods.at(i));

						//calculate the number of cpus and gpus
						ts_diff(work_A[0], span_A[0], numerator);
						ts_diff(modified_period, span_A[0], denominator);

						int CPUs = (int)ceil(numerator / denominator);

						ts_diff(GPU_work[0], GPU_span[0], numerator);
						ts_diff(modified_period, GPU_span[0], denominator);

						int GPUs = (int)ceil(numerator / denominator);

						if (modified_period < GPU_span[0] || modified_period < span_A[0])
							continue;

						//check if this mode is unique
						bool is_unique = true;
						for(size_t j = 0; j < unique_modes.size(); j++){

							if (unique_modes[j].first == CPUs && unique_modes[j].second == GPUs)
								is_unique = false;
						
						}

						if (is_unique){

							struct parsed_task_mode_info mode_info;

							strcpy(mode_info.mode_type, "heavy");

							mode_info.work_sec = work_A[0].tv_sec;
							mode_info.work_nsec = work_A[0].tv_nsec;
							mode_info.span_sec = span_A[0].tv_sec;
							mode_info.span_nsec = span_A[0].tv_nsec;
							mode_info.period_sec = nanosecond_periods.at(i) * 2 / 1000000000;
							mode_info.period_nsec = nanosecond_periods.at(i) * 2 % 1000000000;

							//gpu now
							mode_info.gpu_work_sec = GPU_work[0].tv_sec;		
							mode_info.gpu_work_nsec = GPU_work[0].tv_nsec;
							mode_info.gpu_span_sec = GPU_span[0].tv_sec;
							mode_info.gpu_span_nsec = GPU_span[0].tv_nsec;

							mode_info.gpu_period_sec = nanosecond_periods.at(i) * 2 / 1000000000;
							mode_info.gpu_period_nsec = nanosecond_periods.at(i) * 2 % 1000000000;
						
							unique_modes.push_back({CPUs, GPUs});

							//push the mode
							task_info.modes.push_back(mode_info);

						}

					}

				}
			}


			else if (work_elastic){


				//we want to walk the work_A from min to max and find mappings for all 
				//possible configurations of cpus we can end up with.
				if (is_pure_cpu_task){

					//calc the minimum and maximum cpu allocations
					int min_CPUs = 0;
					int max_CPUs = 0;

					for (int i = 0; i < 2; i++){

						ts_diff(work_A[i], span_A[i], numerator);
						ts_diff(period[i], span_A[i], denominator);

						int CPUs = (int)ceil(numerator / denominator);

						if (CPUs > max_CPUs)
							max_CPUs = CPUs;

						if (CPUs < min_CPUs)
							min_CPUs = CPUs;

					}

					//set denominator again just so it's easier to see
					//when looking back at this code
					ts_diff(period[0], span_A[0], denominator);

					//now we just solve for the periods which 
					//can make the cpu allocations which fall
					//between min and max
					for (int i = min_CPUs; i <= max_CPUs; i++){

						//determine the period
						//by solving for the period
						//which can give us the desired
						//CPU allocation
						long long denominator_as_value = get_timespec_in_ns(denominator);
						denominator_as_value *= i;

						//get the span_A as a value
						long long span_as_value = get_timespec_in_ns(span_A[0]);

						long long target_work = denominator_as_value - span_as_value;

						//make a mode out of this
						struct parsed_task_mode_info mode_info;

						strcpy(mode_info.mode_type, "heavy");

						mode_info.work_sec = target_work / 1000000000;
						mode_info.work_nsec = target_work % 1000000000;

						mode_info.span_sec = span_A[0].tv_sec;
						mode_info.span_nsec = span_A[0].tv_nsec;

						mode_info.period_sec = period[0].tv_sec;
						mode_info.period_nsec = period[0].tv_nsec;

						//push the mode
						task_info.modes.push_back(mode_info);

					}

				}

				//if not a pure cpu task, then we do the same thing for both 
				//the cpu and the gpu resources with the modified period
				else{

					std::vector<std::pair<int, long>> allocations_and_work_cpu;
					std::vector<std::pair<int, long>> allocations_and_work_gpu;

					//calc the minimum and maximum processor allocations
					int min_CPUs = 100000;
					int max_CPUs = 0;

					int min_GPUs = 100000;
					int max_GPUs = 0;

					for (int i = 0; i < 2; i++){

						auto modified_period = period[i] * 0.5;

						ts_diff(work_A[i], span_A[i], numerator);
						ts_diff(modified_period, span_A[i], denominator);

						int CPUs = (int)ceil(numerator / denominator);

						if (CPUs > max_CPUs)
							max_CPUs = CPUs;

						if (CPUs < min_CPUs)
							min_CPUs = CPUs;

						ts_diff(GPU_work[i], GPU_span[i], numerator);
						ts_diff(modified_period, GPU_span[i], denominator);

						int GPUs = (int)ceil(numerator / denominator);

						if (GPUs > max_GPUs)
							max_GPUs = GPUs;

						if (GPUs < min_GPUs)
							min_GPUs = GPUs;

					}

					//set numerator again just so it's easier to see
					//when looking back at this code
					auto modified_period_here = period[0] * 0.5;
					ts_diff(modified_period_here, span_A[0], denominator);

					//now we just solve for the periods which 
					//can make the cpu allocations which fall
					//between min and max
					for (int i = min_CPUs; i <= max_CPUs; i++){

						//determine the period
						//by solving for the period
						//which can give us the desired
						//CPU allocation
						long long denominator_as_value = get_timespec_in_ns(denominator);
						denominator_as_value *= i;

						//get the span_A as a value
						long long span_as_value = get_timespec_in_ns(span_A[0]);

						long long target_work = denominator_as_value - span_as_value;

						//add it to the candidate modes
						allocations_and_work_cpu.push_back({i, target_work});

					}

					//do the same procedure but now to get all the possible GPU allocations
					ts_diff(modified_period_here, GPU_span[0], denominator);

					for (int i = min_GPUs; i <= max_GPUs; i++){

						//determine the period
						//by solving for the period
						//which can give us the desired
						//CPU allocation
						long long denominator_as_value = get_timespec_in_ns(denominator);
						denominator_as_value *= i;

						//get the span_A as a value
						long long span_as_value = get_timespec_in_ns(GPU_span[0]);

						long long target_work = denominator_as_value - span_as_value;

						//add it to the candidate modes
						allocations_and_work_gpu.push_back({i, target_work});

					}

					//now loop over all the candidate pairs, and combine them into
					//unqiue modes
					for (size_t i = 0; i < allocations_and_work_cpu.size(); i++){

						for (size_t j = 0; j < allocations_and_work_gpu.size(); j++){

							struct parsed_task_mode_info mode_info;

							strcpy(mode_info.mode_type, "heavy");

							mode_info.work_sec = allocations_and_work_cpu[i].second / 1000000000;
							mode_info.work_nsec = allocations_and_work_cpu[i].second % 1000000000;

							mode_info.span_sec = span_A[0].tv_sec;
							mode_info.span_nsec = span_A[0].tv_nsec;

							mode_info.period_sec = period[0].tv_sec;
							mode_info.period_nsec = period[0].tv_nsec;

							//gpu now
							mode_info.gpu_work_sec = allocations_and_work_gpu[j].second / 1000000000;	
							mode_info.gpu_work_nsec = allocations_and_work_gpu[j].second % 1000000000;

							mode_info.gpu_span_sec = GPU_span[0].tv_sec;
							mode_info.gpu_span_nsec = GPU_span[0].tv_nsec;

							mode_info.gpu_period_sec = period[0].tv_sec;
							mode_info.gpu_period_nsec = period[0].tv_nsec;

							//push the mode
							task_info.modes.push_back(mode_info);

						}

					}

				}


			}

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

	// Initialize processor topology variables
	int proc_topology_A = 0, proc_topology_B = 1, proc_topology_C = 2, proc_topology_D = 3;
	
	if (read_scheduling_yaml_file(ifs, &schedulable, &parsed_tasks, &sec_to_run, &nsec_to_run, 
		&proc_topology_A, &proc_topology_B, &proc_topology_C, &proc_topology_D) != 0) 
		return RT_GOMP_CLUSTERING_LAUNCHER_FILE_PARSE_ERROR;

	//move all the data to the shared memory
	yaml_object->schedulable = schedulable;
	yaml_object->sec_to_run = sec_to_run;
	yaml_object->nsec_to_run = nsec_to_run;
	
	// Store processor topology configuration
	yaml_object->processor_topology_A = proc_topology_A;
	yaml_object->processor_topology_B = proc_topology_B;
	yaml_object->processor_topology_C = proc_topology_C;
	yaml_object->processor_topology_D = proc_topology_D;
	
	yaml_object->num_tasks_parsed = parsed_tasks.size();

	for (int i = 0; i < (int) parsed_tasks.size(); i++) {

		yaml_object->parsed_tasks[i] = parsed_tasks[i];

		yaml_object->num_modes_parsed[i] = parsed_tasks[i].modes.size();

		for (int j = 0; j < (int) parsed_tasks[i].modes.size(); j++) {

			yaml_object->parsed_modes[i][j] = parsed_tasks[i].modes[j];

		}

	}

	std::cout << "YAML PARSER: Finished parsing the YAML file." << std::endl;

	return 0;
}

