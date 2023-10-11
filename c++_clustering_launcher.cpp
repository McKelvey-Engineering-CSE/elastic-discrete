// Argument: the name of the taskset/schedule file:

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <sys/wait.h>
#include <list>
#include <thread>

#include "single_use_barrier.h"
#include "timespec_functions.h"
#include "scheduler.h"

//using namespace std;

timespec current_time,start_time,run_time,end_time;
timespec cur_time;
timespec vdeadline={0,0};
timespec zero={0,0};

std::string program_name;

// Define the name of the barrier used for synchronizing tasks after creation
const std::string barrier_name = "RT_GOMP_CLUSTERING_BARRIER";
const std::string barrier_name2 = "RT_GOMP_CLUSTERING_BARRIER2";
int ret_val;

bool needs_scheduling = false;
Scheduler * scheduler;
pid_t process_group;

//BOOLEAN VALUE FOR WHETHER WE INITIALLY INCLUDE LAST TASK IN THE SCHEDULE
bool add_last_task = true;
TaskData * last_task;

struct itimerspec disarming_its, virtual_dl_timer_its;
//static timer_t vd_timer;
struct sigevent sev;

int ret;

void scheduler_task()
{
	//stick this thread to CPU 0.	
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(0,&mask);
	pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&mask);

	//Make sure all tasks are ready. Wait at barrier.
	ret_val = await_single_use_barrier(barrier_name.c_str());
	if (ret_val != 0)
	{
		std::cerr << "ERROR: Barrier error for scheduling task.\n";
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
		}
		get_time(&cur_time);
	}

	//std::cerr << "FINISHED!!\n");
}

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

void sigrt0_handler( int signum ){
	needs_scheduling = true;
}
void sigrt1_handler( int signum ){

}

void sigrt2_handler(int signum){
	//scheduler->add_task(last_task->get_min_work().tv_sec, last_task->get_min_work().tv_nsec, last_task->get_max_work().tv_sec, last_task->get_max_work().tv_nsec, last_task->get_span().tv_sec, last_task->get_span().tv_nsec, last_task->get_min_period().tv_sec, last_task->get_min_period().tv_nsec, last_task->get_max_period().tv_sec, last_task->get_max_period().tv_nsec, 0, 0, 0, 0, last_task->get_elasticity());				
	kill(0, SIGRTMIN+0);	
}               

int main(int argc, char *argv[])
{
	process_group = getpgrp();
	std::vector<std::string> args(argv, argv+argc);

	int ret_val;

	//Set up a signal handler for SIGRT0
	void (*ret_handler)(int);
	ret_handler = signal(SIGRTMIN+0, sigrt0_handler);

	if( ret_handler == SIG_ERR ){
		std::cerr << "ERROR: Call to Signal failed, reason: " << strerror(errno) << "\n";
		exit(-1);
	}

	ret_handler = signal(SIGRTMIN+1, sigrt1_handler);
	if( ret_handler == SIG_ERR ){
		std::cerr << "ERROR: Call to Signal failed, reason: " << strerror(errno) << "\n";
		exit(-1);
	}

	// Verify the number of arguments
	if (argc != 2)
	{
		std::cerr << "ERROR: The program must receive a single argument which is the taskset/schedule filename without any extension.\n";
		return RT_GOMP_CLUSTERING_LAUNCHER_ARGUMENT_ERROR;
	}
	
	// Determine the taskset and schedule (.rtps) filename from the program argument
	std::string schedule_filename(args[1] + ".rtps");

	// Make sure we have a .rtps file. Quit if not.
	//struct stat schedule_stat; //struct stat still in from use of legacy .rtpt file when access date was used to create new .rtps files. This is OK.
	//if ((stat(schedule_filename.c_str(), &schedule_stat)) == -1)

	//just use ifstream good property....????
	std::ifstream ifs(schedule_filename);
	if(!ifs.good())
	{
		std::cerr << "ERROR: Cannot find schedule file: " << schedule_filename << "\n";
		return RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR;
	}
	
	// Open the schedule (.rtps) file
	if (!ifs.is_open())
	{
		std::cerr << "ERROR: Cannot open schedule file.\n";
		return RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR;
	}

	bool schedulable;
	unsigned num_tasks=0;
	unsigned sec_to_run=0;
	long nsec_to_run=0;
	std::string line;//, num_tasks_s, num_modes_s, sec_to_run_s, nsec_to_run;

	getline(ifs, line);
	std::istringstream firstline(line);
	if(!(firstline >> schedulable))
	{
		std::cerr << "ERROR: First line of .rtps file error.\n Format: <taskset schedulability>.\n";
		return RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR;
	}
	if(!schedulable)
	{
		std::cerr << "ERROR: Taskset deemed not schedulable by .rtps file. Exiting.\n";
		return RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR;
	}

	getline(ifs, line);
	std::istringstream secondline(line);
	if(!((secondline >> num_tasks) && (secondline >> sec_to_run) && (secondline >> nsec_to_run)))
	{
		std::cerr << "ERROR: Second line of .rtps file error.\n Format: <number of tasks> <s to run> <ns to run>.\n";
		return RT_GOMP_CLUSTERING_LAUNCHER_FILE_OPEN_ERROR;
	}

	scheduler = new Scheduler(num_tasks,(int) std::thread::hardware_concurrency()-1); //Keep CPU 0 for scheduler.
	
	unsigned num_lines=0;
	//Count the number of tasks
	while (getline(ifs, line))
	{       
		if(line.empty())
			{continue;}
		num_lines  += 1;
	}

	if (!(num_lines == 2*num_tasks))
	{
		std::cerr << "ERROR: Found "<< num_lines/2 << " tasks and expected " << num_tasks << ".\n";
		return RT_GOMP_CLUSTERING_LAUNCHER_FILE_PARSE_ERROR;
	}

	// Seek back to the beginning of the file
	ifs.clear();
	ifs.seekg(0, std::ios::beg);
	
	// Initialize a barrier to synchronize the tasks after creation
	ret_val = init_single_use_barrier(barrier_name.c_str(), num_tasks + 1); //Added +1 for scheduler thread. JO 6/22/18
	if (ret_val != 0)
	{
		std::cerr << "ERROR: Failed to initialize barrier.\n";
		return RT_GOMP_CLUSTERING_LAUNCHER_BARRIER_INITIALIZATION_ERROR;
	}

	ret_val = init_single_use_barrier(barrier_name2.c_str(), num_tasks + 1);
	if (ret_val != 0)
	{
		std::cerr << "ERROR: Failed to initialize barrier.\n";
		return RT_GOMP_CLUSTERING_LAUNCHER_BARRIER_INITIALIZATION_ERROR;
	}

	get_time(&current_time);
	run_time={sec_to_run, nsec_to_run};
	start_time.tv_sec = current_time.tv_sec + 5;//Add 5 seconds to start time so all tasks have enough time to finish their init() stages.
	start_time.tv_nsec = current_time.tv_nsec;
	end_time = start_time + run_time;

	// Iterate over the tasks and fork and execv each one
	std::string task_command_line, task_timing_line;//, task_partition_line;
	std::getline(ifs, task_command_line); std::getline(ifs, task_command_line); //Ignore first two lines of file.
	for (unsigned t = 1; t <= num_tasks; ++t)
	{		
		if (std::getline(ifs, task_command_line))
		{
			std::istringstream task_command_stream(task_command_line);
			
			// Add arguments to this vector of strings. This vector will be transformed into
			// a vector of char * before the call to execv by calling c_str() on each string,
			// but storing the strings in a vector is necessary to ensure that the arguments
			// have different memory addresses. If the char * vector is created directly, by
			// reading the arguments into a string and and adding the c_str() to a vector, 
			// then each new argument could overwrite the previous argument since they might
			// be using the same memory address. Using a vector of strings ensures that each
			// argument is copied to its own memory before the next argument is read.
			std::vector<std::string> task_manager_argvector;
			
			// Add the task program name to the argument vector with number of modes as well as start and finish times
			if (task_command_stream >> program_name)
			{
				task_manager_argvector.push_back(program_name);
				task_manager_argvector.push_back(std::to_string(start_time.tv_sec).c_str());
				task_manager_argvector.push_back(std::to_string(start_time.tv_nsec).c_str());
				task_manager_argvector.push_back(std::to_string(end_time.tv_sec).c_str());
				task_manager_argvector.push_back(std::to_string(end_time.tv_nsec).c_str());	
			}
			else
			{
				std::cerr << "ERROR: Program name not provided for task.\n";
				kill(0, SIGTERM);
				return RT_GOMP_CLUSTERING_LAUNCHER_FILE_PARSE_ERROR;
			}

			std::vector <timespec> work;
			std::vector <timespec> span;
			std::vector <timespec> period;
			int num_modes;

			time_t work_sec;
			long work_nsec;
			time_t span_sec; 
			long span_nsec; 
			time_t period_sec;
			long period_nsec;
			double elasticity;
				

			if (std::getline(ifs, task_timing_line))
			{       
				std::istringstream task_timing_stream(task_timing_line);

				//Read in elasticity coefficient and number of modes.
				if((task_timing_stream >> elasticity) && (task_timing_stream >> num_modes))
				{
					//Make sure we have at least 1 mode.
					if(num_modes <= 0)
					{
						std::cerr << "ERROR: Task " << t << " timing data. At least 1 mode of operation is required. Found " << num_modes << ".\n";
						kill(0, SIGTERM);
						return RT_GOMP_CLUSTERING_LAUNCHER_FILE_PARSE_ERROR;
					}
				}
				else
				{
					std::cerr << "ERROR: Task "<< t <<" timing data. Mal-formed elasticity value and/or number of modes.\n";
					kill(0, SIGTERM);
					return RT_GOMP_CLUSTERING_LAUNCHER_FILE_PARSE_ERROR;
				}

				//Read in Work, Span, and Periods for each mode.
				for(int i=0; i<num_modes; i++)
				{
					if((task_timing_stream >> work_sec) && (task_timing_stream >> work_nsec) && (task_timing_stream >> span_sec) && (task_timing_stream >> span_nsec) && 
							(task_timing_stream >> period_sec) && (task_timing_stream >> period_nsec))	
					{
						//Put work, span, period values on vector
						timespec w = {work_sec,work_nsec};
						timespec s = {span_sec,span_nsec};
						timespec p = {period_sec,period_nsec};
						std::cout << w << " " << s << " " << p << std::endl;
						work.push_back(w);
						span.push_back(s);
						period.push_back(p);
					}
					else
					{
						std::cerr << "ERROR: Task " << t << " timing data. Mal-formed work, span, or period in mode " << i<< ".\n";
					}
				}

				TaskData * td;
				td = scheduler->add_task(elasticity, num_modes, work.data(), span.data(), period.data());
				task_manager_argvector.push_back(std::to_string(td->get_index()));
			}
			else	
			{
				std::cerr << "ERROR: Too few timing parameters were provided for task " << program_name.c_str() << ".\n";
				kill(0, SIGTERM);
				return RT_GOMP_CLUSTERING_LAUNCHER_FILE_PARSE_ERROR;
			}

			// Add the barrier name to the argument vector
			task_manager_argvector.push_back(barrier_name);
			
			// Add the task arguments to the argument vector
			task_manager_argvector.push_back(program_name);
			
			std::string task_arg;
			while (task_command_stream >> task_arg)
			{
				task_manager_argvector.push_back(task_arg);
			}
			
			// Create a vector of char * arguments from the vector of string arguments
			std::vector<const char *> task_manager_argv;
			for (std::vector<std::string>::iterator i = task_manager_argvector.begin(); i != task_manager_argvector.end(); ++i)
			{
				task_manager_argv.push_back(i->c_str());
			}
			
			// NULL terminate the argument vector
			task_manager_argv.push_back(NULL);
			//HERE!!!	
			std::cerr << "Forking and execv-ing task " << program_name.c_str() << "\n";
			
			// Fork and execv the task program
			pid_t pid = fork();
			if (pid == 0)
			{
				// Const cast is necessary for type compatibility. Since the strings are
				// not shared, there is no danger in removing the const modifier.
				execv(program_name.c_str(), const_cast<char **>(&task_manager_argv[0]));
				
				// Error if execv returns
				perror("Execv-ing a new task failed.\n");
				kill(0, SIGTERM);
				return RT_GOMP_CLUSTERING_LAUNCHER_FORK_EXECV_ERROR;
			
			}
			else if (pid == -1)
			{
				perror("Forking a new process for task failed,\n");
				kill(0, SIGTERM);
				return RT_GOMP_CLUSTERING_LAUNCHER_FORK_EXECV_ERROR;
			}	
		}
		else
		{
			std::cerr << "ERROR: Please include at least one task.\n";
			kill(0, SIGTERM);
			return RT_GOMP_CLUSTERING_LAUNCHER_FILE_PARSE_ERROR;
		}
	}

	scheduler->do_schedule();

	//std::cerr << "Scheduler reached scheduling barrier.");	
	if ((ret_val = await_single_use_barrier(barrier_name2.c_str())) != 0)
	{
		std::cerr << "ERROR: Barrier error for scheduling task \n";
		kill(0, SIGTERM);
	}

	std::thread t(scheduler_task);
	
	// Close the file
	ifs.close();
	
	std::cerr << "All tasks started.\n";
	
	// Wait until all child processes have terminated
	while (!(wait(NULL) == -1 && errno == ECHILD));
	
	t.join();

	std::cerr << "All tasks finished.\n";
	
	delete scheduler;
	
	return 0;
}

