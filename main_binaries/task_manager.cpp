// Each real time task should be compiled as a separate program and include task_manager.cpp and task.h
// in compilation. The task struct declared in task.h must be defined by the real time task.
#include <stdint.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <thread>
#include <sstream>
#include <vector>
#include <signal.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include "task.h"
#include "schedule.h"
#include "process_barrier.h"
#include "timespec_functions.h"
#include <signal.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <atomic>
#include <list>
#include "include.h"
#include "thread_barrier.h" 
#include <sys/types.h>
#include <sys/syscall.h>
#include <limits.h>
#include <map>
#include "print_module.h"

#include <chrono>
using namespace std::chrono;

extern "C" 
{
#include "dl_syscalls.h"
}

#define gettid() syscall(SYS_gettid)

#ifdef TRACING
FILE * fd;
#endif

//testing capturing faults and notifying before simply dying
void exit_on_signal(int sig){
	print_module::print(std::cerr, "Signal captured " , strsignal(sig) , ". Program cannot continue. Exiting.\n");
	kill(getppid(), 1);
	exit(-1);
}

//task and cpu params
extern const int NUMCPUS;
extern const int MAXTASKS;

//There are one trillion nanoseconds in a second, or one with nine zeroes
const unsigned nsec_in_sec = 1000000000; 

//The priority that we finalize the program with
const unsigned FINALIZE_PRIORITY = 1;

//The priority that we use when sleeping.
const unsigned SLEEP_PRIORITY = 97;

//The priority that we use during normal execution (configurable by user)
unsigned EXEC_PRIORITY = 7;

//These variables are declared extern in task.h, but need to be
//visible in both places
int futex_val;
thread_barrier bar;
bool missed_dl = false;
int practical_max_cpus;
volatile int total_remain __attribute__ (( aligned (64) ));
double percentile = 1.0;

//This value is used as a return value for system calls
int ret_val;

//The process group ID is used to notify other tasks in this taskset that
//it is time to switch to high criticality mode
pid_t process_group;
pid_t mypid;

cpu_set_t global_cpuset;
struct sched_param global_param;

enum rt_gomp_task_manager_error_codes
{ 
	RT_GOMP_TASK_MANAGER_SUCCESS,
	RT_GOMP_TASK_MANAGER_CORE_BIND_ERROR,
	RT_GOMP_TASK_MANAGER_SET_PRIORITY_ERROR,
	RT_GOMP_TASK_MANAGER_INIT_TASK_ERROR,
	RT_GOMP_TASK_MANAGER_RUN_TASK_ERROR,
	RT_GOMP_TASK_MANAGER_BARRIER_ERROR,
	RT_GOMP_TASK_MANAGER_BAD_DEADLINE_ERROR,
	RT_GOMP_TASK_MANAGER_ARG_PARSE_ERROR,
	RT_GOMP_TASK_MANAGER_ARG_COUNT_ERROR
};

time_t start_sec = 0, end_sec = 0;
long start_nsec = 0, end_nsec = 0;
timespec start_time, end_time, current_period, current_work, current_span, deadline;
int task_index = -1;
int current_mode = -1;
unsigned num_iters = 0;

// Let scheduler know we need a rescheduling.
// Do this by sending signal SIGRTMIN+0.
void initiate_reschedule(){

	killpg(process_group, SIGRTMIN + 0);

}

cpu_set_t current_cpu_mask;
struct sched_param sp;
bool active_threads[64];
bool needs_reschedule = false;

std::mutex con_mut;
Schedule schedule(std::string("EFSschedule"));

//has pthread_t at omp_thread index
std::vector<pthread_t> threads;

//tells which thread is on a CPU
std::map<int, pthread_t> thread_at_cpu;

//tells OMP number of thread
std::map<pthread_t, int> omp_thread_index;

void sigrt0_handler(int signum){}

void reschedule();

void sigrt1_handler(int signum){
	needs_reschedule = true;
}

void modify_self(int new_mode){

	schedule.get_task(task_index)->set_current_mode(new_mode, true);
	killpg(process_group, SIGRTMIN+0);

}

void allow_change(){

	schedule.get_task(task_index)->reset_changeable();
	killpg(process_group, SIGRTMIN+0);

}

//function to either pass off resources or take them from other tasks
void reschedule(){

	// Set up everything to begin as scheduled.
    current_period = schedule.get_task(task_index)->get_current_period();
    current_work = schedule.get_task(task_index)->get_current_work();
	current_mode = schedule.get_task(task_index)->get_current_mode();
	deadline = current_period;
	percentile = schedule.get_task(task_index)->get_percentage_workload();

	for (int selected_task = 0; selected_task < schedule.count(); selected_task++){

		for (int cpu_under_consideration = 1; cpu_under_consideration <= NUMCPUS; cpu_under_consideration++){

			//if we are giving up a CPU and making it passive
			if (schedule.get_task(task_index)->transfers(selected_task, cpu_under_consideration)){

				//mark the CPU as passive
				schedule.get_task(task_index)->clr_active_cpu(cpu_under_consideration);
				schedule.get_task(task_index)->set_passive_cpu(cpu_under_consideration);

				//remove it from our active table...
				//FIXME: These two tables should not be separate when switching between passive and active... just combine them
				active_threads[omp_thread_index[thread_at_cpu[cpu_under_consideration]]] = false;

				//set thread priority to sleep
				global_param.sched_priority = SLEEP_PRIORITY;
	            ret_val = pthread_setschedparam(thread_at_cpu[cpu_under_consideration], SCHED_RR, &global_param);

				if (ret_val < 0)
					print_module::print(std::cerr, "ERROR: could not set sleep priority when making thread passive.\n");
				
			}

			//if another task has given up a CPU for us to receive and mark active
			else if (schedule.get_task(task_index)->receives(selected_task, cpu_under_consideration)){

				//if this cpu was already one of our passive threads
				if (schedule.get_task(task_index)->get_passive_cpu(cpu_under_consideration)){

					//mark it as active
					schedule.get_task(task_index)->clr_passive_cpu(cpu_under_consideration);
					schedule.get_task(task_index)->set_active_cpu(cpu_under_consideration);

					//add it to our active table
					//FIXME: SEE ABOVE
					active_threads[omp_thread_index[thread_at_cpu[cpu_under_consideration]]] = true;

					//set thread priority to active
					global_param.sched_priority = EXEC_PRIORITY;
					ret_val = pthread_setschedparam(thread_at_cpu[cpu_under_consideration], SCHED_RR, &global_param);

					
				}

				//if this cpu is new to us
				else {

					//go through all CPUS 
					for (int alternate_cpu = NUMCPUS; alternate_cpu >= 1; alternate_cpu--) {

						//until we find one that is in our passive set
						if (schedule.get_task(task_index)->get_passive_cpu(alternate_cpu)){
							
							//your guess is as good as mine as to why we print this
							for (int i = 1; i <= NUMCPUS; i++)
								print_module::print(std::cout, active_threads[i] , "\n");
							print_module::print(std::cout, omp_thread_index[thread_at_cpu[alternate_cpu]], " ", thread_at_cpu[alternate_cpu], " ",  alternate_cpu, "\n");		

							//get the handle of the thread that is at cpu "alternate_cpu"
							//that handle can be turned into an index via omp_thread_index
							//using that index, set the thread to active
							active_threads[omp_thread_index[thread_at_cpu[alternate_cpu]]] = true;

							//set the corresponding CPU to not passive
							schedule.get_task(task_index)->clr_passive_cpu(alternate_cpu);

							//the thread that was at "cpu_under_consideration" is now at "alternate_cpu"
							//make a second entry into the map to reflect that
							//so all queries to "cpu_under_consideration" will return the handle of the thread that was at "alternate_cpu"
							thread_at_cpu[cpu_under_consideration] = thread_at_cpu[alternate_cpu];

							//remove old entry
							thread_at_cpu.erase(alternate_cpu);

							//this is confusing.........
							//we already set the table to active above at the index "alternate_cpu"
							//which is now the same handle as "cpu_under_consideration"
							//active_threads[omp_thread_index[thread_at_cpu[cpu_under_consideration]]] = true;

							//set the CPU to active
							schedule.get_task(task_index)->set_active_cpu(cpu_under_consideration); 

							//set the thread to active
							global_param.sched_priority = 7;
							ret_val = pthread_setschedparam(thread_at_cpu[cpu_under_consideration], SCHED_RR, &global_param);	

							//update our CPU mask
							//to show the gained "cpu_under_consideration" CPU
							CPU_ZERO(&global_cpuset);
							CPU_SET(cpu_under_consideration, &global_cpuset);

							pthread_setaffinity_np(thread_at_cpu[cpu_under_consideration], sizeof(cpu_set_t), &global_cpuset);

							break;
						}
					}
				}
			}	
		}
	}		

	bar.mc_bar_reinit(schedule.get_task(task_index)->get_current_CPUs());		

}

bool execution_condition(timespec current_time, timespec end_time, int iterations, int current_iterations)
{
	if (end_time.tv_sec != 0 && end_time.tv_nsec != 0 && current_time >= end_time) {
		return false;
	}
	return current_iterations != iterations;
};

int main(int argc, char *argv[])
{
	fflush(stdout);
	fflush(stderr);

	//setup the capture handlers
	signal(SIGSEGV, exit_on_signal);
	signal(SIGTERM, exit_on_signal);
	signal(SIGABRT, exit_on_signal);
	signal(SIGILL, exit_on_signal);
	signal(SIGFPE, exit_on_signal);

	#ifdef TRACING
	fd = fopen( "/sys/kernel/debug/tracing/trace_marker", "a" );
	
	if (fd == NULL){

		std::perror("Error: TRACING is defined and you are not using trace-cmd.");
		return -1;

	}
	#endif

	std::string command_string;	
	for (int i = 0; i < argc; i++)
		command_string += std::string(argv[i]) + " ";

	//Get our own PID and store it
	mypid = getpid();

	//Determine what our current process group is, so we can notify the
	//system if we miss our virtual deadline
	process_group = getpgrp();

	//Set up a signal handler for SIGRT0, this manages the notification of
	//the system high crit transition
	void (*ret_handler)(int);
	ret_handler = signal(SIGRTMIN + 0, sigrt0_handler);

	if (ret_handler == SIG_ERR){

		print_module::print(std::cerr,  "ERROR: Call to Signal failed, reason: " , strerror(errno) , "\n");
		exit(-1);

	}

	ret_handler = signal(SIGRTMIN + 1, sigrt1_handler);
	if (ret_handler == SIG_ERR){

		print_module::print(std::cerr,  "ERROR: Call to Signal failed, reason: " , strerror(errno) , "\n");
		exit(-1);

	}

	// Process command line arguments	
	const char *task_name = argv[0];

	int iterations = 0;
	
	if(!((std::istringstream(argv[1]) >> start_sec) &&
		(std::istringstream(argv[2]) >> start_nsec) &&
		(std::istringstream(argv[3]) >> iterations) &&
		(std::istringstream(argv[4]) >> end_sec) &&
        (std::istringstream(argv[5]) >> end_nsec) &&
		(std::istringstream(argv[6]) >> EXEC_PRIORITY) &&
		(std::istringstream(argv[7]) >> iindex)))
	{
		print_module::print(std::cerr,  "ERROR: Cannot parse input argument for task " , task_name , "\n");
		kill(0, SIGTERM);
		return RT_GOMP_TASK_MANAGER_ARG_PARSE_ERROR;
	
	}
	
	start_time = {start_sec, start_nsec};
	end_time = {end_sec, end_nsec};
	
	char *barrier_name = argv[8];
	int task_argc = argc - 9;                                             
	char **task_argv = &argv[9];

	//Wait at barrier for the other tasks but mainly to make sure scheduler has finished
	if ((ret_val = process_barrier::await_and_destroy_barrier("BAR_2")) != 0){

		print_module::print(std::cerr,  "ERROR: Barrier error for task " , task_name , "\n");
		kill(0, SIGTERM);
		return RT_GOMP_TASK_MANAGER_BARRIER_ERROR;

	}

	// Set up everything to begin as scheduled.
	current_work = schedule.get_task(task_index)->get_current_work();
	current_period = schedule.get_task(task_index)->get_current_period();
	deadline = current_period;
	percentile = schedule.get_task(task_index)->get_percentage_workload();

	CPU_ZERO(&current_cpu_mask);
	
	//add each cpu from our min up to our cpu mask
	for (int i = 0; i < schedule.get_task(task_index)->get_current_CPUs(); i++ )
		CPU_SET(schedule.get_task(task_index)->get_current_lowest_CPU() + i, &current_cpu_mask);
	
	std::lock_guard<std::mutex> lk(con_mut);

	//print task information
	std::ostringstream task_info;
	std::string task_header = "|  Task " + std::to_string(task_index) + " (PID: " + std::to_string(getpid()) + ") |\n";
	size_t blank_size = task_header.size();

	print_module::buffered_print(task_info, "\n", std::string(blank_size - 1, '='), "\n");
	print_module::buffered_print(task_info, task_header);
	print_module::buffered_print(task_info, std::string(blank_size - 1, '='), "\n\n");

	//cpu info
	print_module::buffered_print(task_info, "CPU Metrics: \n");
	print_module::buffered_print(task_info, "	- Lowest CPU: ", schedule.get_task(task_index)->get_current_lowest_CPU(), "\n");
	print_module::buffered_print(task_info, "	- Current CPUs: ", schedule.get_task(task_index)->get_current_CPUs(), "\n");
	print_module::buffered_print(task_info, "	- Minimum CPUs: ", schedule.get_task(task_index)->get_min_CPUs(), "\n");
	print_module::buffered_print(task_info, "	- Maximum CPUs: ", schedule.get_task(task_index)->get_max_CPUs(), "\n");
	print_module::buffered_print(task_info, "	- Practical Max: ", schedule.get_task(task_index)->get_practical_max_CPUs(), "\n\n");

	//gpu info
	print_module::buffered_print(task_info, "GPU Metrics: \n");
	print_module::buffered_print(task_info, "	- Lowest GPU: ", schedule.get_task(task_index)->get_current_lowest_GPU(), "\n");
	print_module::buffered_print(task_info, "	- Current GPUs: ", schedule.get_task(task_index)->get_current_GPUs(), "\n");
	print_module::buffered_print(task_info, "	- Minimum GPUs: ", schedule.get_task(task_index)->get_min_GPUs(), "\n");
	print_module::buffered_print(task_info, "	- Maximum GPUs: ", schedule.get_task(task_index)->get_max_GPUs(), "\n");
	print_module::buffered_print(task_info, "	- Practical Max: ", schedule.get_task(task_index)->get_practical_max_GPUs(), "\n\n");

	//timing info
	print_module::buffered_print(task_info, "Timing Metrics: \n");
	print_module::buffered_print(task_info, "	- Period s: ", schedule.get_task(task_index)->get_current_period().tv_sec , " s\n");
	print_module::buffered_print(task_info, "	- Period ns: ", schedule.get_task(task_index)->get_current_period().tv_nsec , " ns\n\n");

	struct sched_param param;

	param.sched_priority = EXEC_PRIORITY;
	ret_val = sched_setscheduler(getpid(), SCHED_RR, &param);

	if (ret_val != 0)
 		print_module::print(std::cerr,  "WARNING: " , getpid() , " Could not set priority. Returned: " , errno , "  (" , strerror(errno) , ")\n");

	// OMP settings
	omp_set_nested(0);
	omp_set_dynamic(0);
	omp_set_schedule(omp_sched_dynamic, 1);	

	practical_max_cpus = schedule.get_task(task_index)->get_practical_max_CPUs();
	omp_set_num_threads(practical_max_cpus);

	for (int t = 1; t <= NUMCPUS; t++){

		schedule.get_task(task_index)->clr_active_cpu(t);
		schedule.get_task(task_index)->clr_passive_cpu(t);
	
	}

	for (unsigned int j = 0; j < std::thread::hardware_concurrency(); j++)
		active_threads[j] = false;
 
	//pretty sure this is a war crime
	//(translation for those reading this later... each thread as specified by OMP 
	//gets activated right here and stores it's pthread handle into the "leftmost" 
	//portion of the double map association structure)
	threads.reserve(NUMCPUS);
	#pragma omp parallel
	{

		threads[omp_get_thread_num()] = pthread_self();
		omp_thread_index[pthread_self()] = omp_get_thread_num();

	}

	//Determine which threads are active and passive. Pin, etc.
	std::string active_cpu_string = "  ";
	std::string passive_cpu_string = "  ";

	//for CPUs that we MAY have at some point (practical max)
	for (int j = 0; j < practical_max_cpus; j++){

		//Mark all cpus from our minimum to our minimum + the number of CPUs we are using as active
		active_threads[j] = (j < schedule.get_task(task_index)->get_current_CPUs());

		//"j" does not map to CPU index, so we need to calculate it
		int p = (schedule.get_task(task_index)->get_current_lowest_CPU() + j - 1) % (NUMCPUS) + 1;

		//Our first CPU is our permanent CPU 
		if (j == 0)
			schedule.get_task(task_index)->set_permanent_CPU(p);

		if (active_threads[j]){

			global_param.sched_priority = EXEC_PRIORITY;

			schedule.get_task(task_index)->set_active_cpu(p);

			active_cpu_string += std::to_string(p) + ", ";

		}

		else {

			global_param.sched_priority = SLEEP_PRIORITY;

			schedule.get_task(task_index)->set_passive_cpu(p);

			passive_cpu_string += std::to_string(p) + ", ";

		}
		
		//set our thread at "j" to be at CPU "p" via our thread location map
		ret_val = pthread_setschedparam(threads[j], SCHED_RR, &global_param);
		thread_at_cpu[p] = threads[j];

		CPU_ZERO(&global_cpuset);
		CPU_SET(p, &global_cpuset);

		pthread_setaffinity_np(threads[j], sizeof(cpu_set_t), &global_cpuset);
 
  	}

	//print active vs passive CPUs
	print_module::buffered_print(task_info, "CPU Core Configuration: \n");
	print_module::buffered_print(task_info, "	- Active:", active_cpu_string.substr(0, active_cpu_string.size() - 2), "\n");
	print_module::buffered_print(task_info, "	- Passive:", passive_cpu_string.substr(0, active_cpu_string.size() - 2), "\n\n");

	//command line params
	print_module::buffered_print(task_info, "Command Line Parameters: \n");
	print_module::buffered_print(task_info, "	- Args: ", command_string, "\n\n");

	//flush all task info to terminal
	print_module::flush(std::cerr, task_info);

	//Initialize the program barrier
	bar.mc_bar_init(schedule.get_task(task_index)->get_current_CPUs());

	#ifdef PER_PERIOD_VERBOSE
		std::vector<uint64_t> period_timings;
	#endif

	// Initialize the task
	if (schedule.get_task(task_index) && task.init != NULL){

		if (task.init(task_argc, task_argv) != 0){

			print_module::print(std::cerr, "ERROR: Task initialization failed for task ", task_name, "\n");
			kill(0, SIGTERM);
			return RT_GOMP_TASK_MANAGER_INIT_TASK_ERROR;
		
		}

	}
	
	// Wait at barrier for the other tasks
	if ((ret_val = process_barrier::await_and_destroy_barrier(barrier_name)) != 0){

		print_module::print(std::cerr, "ERROR: Barrier error for task ", task_name, "\n");
		kill(0, SIGTERM);
		return RT_GOMP_TASK_MANAGER_BARRIER_ERROR;
	
	}
	
	//Don't go until it's time.
	//Grab a timestamp at the start of real-time operation
	timespec current_time;
	do { 
		get_time(&current_time); 
	} while(current_time < start_time);

	// Initialize timing controls
	unsigned deadlines_missed = 0;
	timespec correct_period_start, actual_period_start, period_finish, period_runtime;
	get_time(&correct_period_start);
	timespec max_period_runtime = { 0, 0 };
	uint64_t total_nsec = 0;

	while(execution_condition(current_time, end_time, iterations, num_iters)){

		if (schedule.get_task(task_index))

			num_iters++;

		// Sleep until the start of the period
		sleep_until_ts(correct_period_start);

        #ifdef TRACING
			fprintf( fd, "thread %ld: starting iteration %d\n", gettid() ,num_iters);
			fflush( fd );
        #endif

		//when we start doing the work
   		get_time(&actual_period_start);

		// Reset the awaited count of threads before every period
		__atomic_store_n( &total_remain, practical_max_cpus, __ATOMIC_RELEASE ); //JAMES ORIGINAL 10/3/17

		if (schedule.get_task(task_index))
			ret_val = task.run(task_argc, task_argv);

		//Get the finishing time of the current period
		get_time(&period_finish);
		if (ret_val != 0){
			
			print_module::print(std::cerr,  "ERROR: Task run failed for task " , task_name, "\n");
			return RT_GOMP_TASK_MANAGER_RUN_TASK_ERROR;

		}
		
		// Check if the task finished before its deadline and record the maximum running time
		ts_diff(correct_period_start, period_finish, period_runtime);

		if (period_runtime > deadline) {

			deadlines_missed += 1;
			missed_dl = true;

			#ifdef TRACING
				fprintf( fd, "thread %d: missed deadline iteration %d\n", getpid() ,num_iters);
				fflush( fd );
            #endif
		
		}
		
		else {

			missed_dl = false;
		
		}

		if (period_runtime > max_period_runtime) max_period_runtime = period_runtime;
		total_nsec += period_runtime.tv_nsec + nsec_in_sec * period_runtime.tv_sec;

		// Update the period_start time
		correct_period_start = correct_period_start + current_period;

		if (needs_reschedule){

			bool ready = true;

			//Check other tasks to see if this task can transition yet. It can if it is giving up a CPU, or is gaining a CPU that has been given up.
			for (int other_task = 0; other_task < schedule.count(); other_task++)
				for (int cpu_being_considered = 1; cpu_being_considered <= NUMCPUS; cpu_being_considered++)
					if ((schedule.get_task(other_task)->transfers(task_index, cpu_being_considered)))
						if ((schedule.get_task(other_task))->get_num_adaptations() <=  (schedule.get_task(task_index))->get_num_adaptations())
							ready = false;

			if (ready){

				#ifdef TRACING
					fprintf(fd, "thread %d: starting reschedule\n", getpid());
					fflush(fd);
				#endif

				reschedule();
				
				print_module::task_print(std::cerr, "System: finished reschedule\n");	

				#ifdef TRACING
					fprintf(fd, "thread %d: finished reschedule\n", getpid());
					fflush(fd);
				#endif

				schedule.get_task(task_index)->set_num_adaptations(schedule.get_task(task_index)->get_num_adaptations() + 1);
				needs_reschedule = false;
			}

			else{

				//Gaining a processor that wasn't ready yet.
				print_module::print(std::cerr,  "task ", getpid(), " can't reschedule!\n");
			
			}
		}

		get_time(&current_time);
		
	}

	// Lower priority as soon as we're done running in real-time
	sp.sched_priority = FINALIZE_PRIORITY;
	ret_val = sched_setscheduler(getpid(), SCHED_RR, &sp);
	
	if (ret_val != 0)
		std::perror("WARNING: Could not set FINALIZE_PRIORITY");
	
	#ifdef TRACING
		fclose(fd);
	#endif

	// Finalize the task
	if (schedule.get_task(task_index) && task.finalize != NULL){

		ret_val = task.finalize(task_argc, task_argv);
		if (ret_val != 0)
			print_module::print(std::cerr,   "WARNING: Task finalization failed for task " , task_name , "\n");
		
	}

	//Print out useful information.
	std::ostringstream task_output;
	print_module::buffered_print(task_output,  "\n(" , mypid , ") Deadlines missed for task " , task_name , ": " , deadlines_missed , "/" , num_iters , "\n");
	print_module::buffered_print(task_output,  "(" , mypid , ") Max running time for task " , task_name , ": " , (int)max_period_runtime.tv_sec , " sec  " , max_period_runtime.tv_nsec , " nsec\n");
	print_module::buffered_print(task_output,  "(" , mypid , ") Avg running time for task " , task_name , ": " , (total_nsec / (num_iters)) / 1000000.0 , " msec\n");
	print_module::buffered_print(task_output, deadlines_missed, " ", num_iters, " ", omp_get_num_threads(), " ", max_period_runtime, "\n\n");
	print_module::flush(std::cerr, task_output);

	#ifdef PER_PERIOD_VERBOSE
		std::ofstream outfile("time_results.txt", std::ios::out | std::ios::app);

		print_module::print(outfile, omp_get_num_procs());

		for (auto period_timing : period_timings)
			print_module::print(outfile, " ", period_timing);
		
		print_module::print(outfile, "\n");
		outfile.close();
	#endif

	fflush(stdout);
	fflush(stderr);

return 0;
}	
