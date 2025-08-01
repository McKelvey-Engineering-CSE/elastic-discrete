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

#ifdef OMP_OVERRIDE

	#include "omp_replacement.hpp"

	//-1 means use all available processors
	ThreadPool<> omp(-1);

#endif

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
volatile int total_remain __attribute__ (( aligned (64) ));
double percentile = 1.0;

int current_threads_active = 0;

//This value is used as a return value for system calls
int ret_val;

//if we set this then we will need a barrier for explicit sync
bool explicit_sync = false;

//The process group ID is used to notify other tasks in this taskset that
//it is time to switch to high criticality mode
pid_t process_group;
pid_t mypid;

//variable to track whether or not the mode change we requested as actually finished
bool mode_change_finished = true;

//buffer to share with james simply for vanity
std::ostringstream reschedule_buffer;

cpu_set_t global_cpuset;
struct sched_param global_param;

//omp replacement thread pool
__uint128_t processor_A_mask;
__uint128_t processor_B_mask;
__uint128_t processor_C_mask;
__uint128_t processor_D_mask;

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

struct sched_param sp;
bool active_threads[64];
bool needs_reschedule = false;

int* shared_array = NULL;
int shm_fd = -1;

std::mutex con_mut;
Schedule schedule(std::string("EFS"), false);

//has pthread_t at omp_thread index
std::vector<pthread_t> threads;

//tells which thread is on a CPU
std::map<int, pthread_t> thread_at_cpu;

//tells OMP number of thread
std::map<pthread_t, int> omp_thread_index;

//name of the memory segment which controls 
//whether or not a tas's request can be granted
//due to a reschedule currently happening
const std::string reschedule_lock = "SCHEDLK";
p_mutex* rescheduling_lock;

//vector representing pool of threads associated with
//this task that are sleeping on cores we gave up
std::vector<pthread_t> thread_handles(NUM_PROCESSOR_A, pthread_t());

void sigrt0_handler(int signum){}

void reschedule();

void sigrt1_handler(int signum){
	needs_reschedule = true;
}

//FIXME: THE TASK DOESN'T KNOW WHAT MODE IT ACTUALLY WANTS, IT JUST KNOWS
//IT NEEDS ONE OF THE MODES WHICH IS DESCRIBED BY THE POSSIBLE MODES WHICH
//ARE MAPPED TO THAT ONE REAL MODE.
bool modify_self(int new_mode){

	if (new_mode == schedule.get_task(task_index)->get_real_current_mode())
		return false;

	if (rescheduling_lock->try_lock() == 0){
		
		mode_change_finished = false;
		schedule.get_task(task_index)->set_real_current_mode(new_mode, true);
		killpg(process_group, SIGRTMIN+0);

		rescheduling_lock->unlock();

		return true;

	}

	else
		return false;

}

int get_current_mode(){

	return schedule.get_task(task_index)->get_real_current_mode();

}

void mimic_simulator(int task_index){

	TaskData * td = schedule.get_task(task_index);

	int task_current_mode = td->get_real_mode(td->get_current_virtual_mode());

	int mode_moving_to = ((task_current_mode + 1) % td->get_original_modes_passed());

	if (modify_self(mode_moving_to))
		pm::print(std::cerr, "Task ", task_index, " is instigating a reschedule to mode ", mode_moving_to, " from mode ", task_current_mode, "\n");

}

void set_cooperative(bool value){

	schedule.get_task(task_index)->set_cooperative(value);

}

void allow_change(){

	schedule.get_task(task_index)->reset_changeable();
	killpg(process_group, SIGRTMIN+0);

}

void set_active_threads(std::vector<int> thread_ids){

    const size_t ARRAY_SIZE = 128;
    char shm_name[32];
    
    // Create process-specific name
    snprintf(shm_name, sizeof(shm_name), "/omp_%d", (int)getpid());
    
    // Create new shared memory segment with create and exclusive flags
    shm_fd = shm_open(shm_name, O_RDWR | O_CREAT, 0600);
    if (shm_fd == -1) {
        std::cerr << "Failed to create new shared memory segment (might already exist): " << shm_name << std::endl;
        return;
    }
    
    // Set the size of the shared memory segment
    if (ftruncate(shm_fd, ARRAY_SIZE * sizeof(int)) == -1) {
        std::cerr << "Failed to set shared memory size" << std::endl;
        close(shm_fd);
        shm_unlink(shm_name);
        return;
    }
    
    // Map the shared memory segment
    shared_array = (int*)mmap(NULL, ARRAY_SIZE * sizeof(int), 
                                 PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shared_array == MAP_FAILED) {
        std::cerr << "Failed to map shared memory" << std::endl;
        close(shm_fd);
        shm_unlink(shm_name);
        return;
    }
    
    //add the thread indices we want to see run
    for (int i = 0; i < (int) thread_ids.size(); i++) 
        shared_array[i + 1] = thread_ids.at(i);
}

//function of pure vanity courtesy of claude
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

//function to either pass off resources or take them from other tasks
void reschedule(){

	// Set up everything to begin as scheduled.
    current_period = schedule.get_task(task_index)->get_current_period();
    current_work = schedule.get_task(task_index)->get_current_work();
	current_mode = schedule.get_task(task_index)->get_current_virtual_mode();
	deadline = current_period;
	percentile = schedule.get_task(task_index)->get_percentage_workload();

	//update our cpu mask
	processor_A_mask = schedule.get_task(task_index)->get_processor_A_mask();

	#ifdef OMP_OVERRIDE
		omp.set_override_mask(processor_A_mask);
	#else
		set_active_threads(schedule.get_task(task_index)->get_processor_A_owned_by_process());
	#endif
		
	//update processor B mask and call update function
	processor_B_mask = schedule.get_task(task_index)->get_processor_B_mask();
	task.update_core_B(processor_B_mask);

	//update processor C mask and call update function
	processor_C_mask = schedule.get_task(task_index)->get_processor_C_mask();
	task.update_core_C(processor_C_mask);

	//update processor D mask and call update function
	processor_D_mask = schedule.get_task(task_index)->get_processor_D_mask();
	task.update_core_D(processor_D_mask);

	//all pretty printing crap
	#ifdef PRETTY_PRINTING

		print_module::buffered_print(reschedule_buffer, "\nTask ", task_index, " finished reschedule:\n");	

		//core A
		print_module::buffered_print(reschedule_buffer, "Core A Owned: [ ");

		std::bitset<128> cpu_mask(processor_A_mask);
		for (int i = 0 ; i < 128; i++)
			print_module::buffered_print(reschedule_buffer, cpu_mask.test(i) ? std::to_string(i) + " " : "");

		print_module::buffered_print(reschedule_buffer, "]\n");

		//core B
		print_module::buffered_print(reschedule_buffer, "Core B Owned: [ ");

		std::bitset<128> gpu_mask(processor_B_mask);
		for (int i = 0 ; i < 128; i++)
			print_module::buffered_print(reschedule_buffer, gpu_mask.test(i) ? std::to_string(i) + " " : "");

		print_module::buffered_print(reschedule_buffer, "]\n");

		//core C
		print_module::buffered_print(reschedule_buffer, "Core C Owned: [ ");

		std::bitset<128> cpu_C_mask(processor_C_mask);
		for (int i = 0 ; i < 128; i++)
			print_module::buffered_print(reschedule_buffer, cpu_C_mask.test(i) ? std::to_string(i) + " " : "");

		print_module::buffered_print(reschedule_buffer, "]\n");

		//core D
		print_module::buffered_print(reschedule_buffer, "Core D Owned: [ ");

		std::bitset<128> gpu_D_mask(processor_D_mask);
		for (int i = 0 ; i < 128; i++)
			print_module::buffered_print(reschedule_buffer, gpu_D_mask.test(i) ? std::to_string(i) + " " : "");

		print_module::buffered_print(reschedule_buffer, "]\n");

		print_module::flush(std::cerr, reschedule_buffer);

	#endif

	//sync all threads
	bar.mc_bar_reinit(schedule.get_task(task_index)->get_current_processors_A());	

	//wait at barrier for all other tasks if we have to
	if (explicit_sync){

		if (process_barrier::await_and_rearm_barrier("EX_SYNC") != 0){

			print_module::print(std::cerr, "ERROR: Barrier error for task ", task_index, "\n");
			kill(0, SIGTERM);
			exit(-1);

		}

	}	

	//set the completion bool
	mode_change_finished = true;

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
		(std::istringstream(argv[7]) >> task_index)))
	{
		print_module::print(std::cerr,  "ERROR: Cannot parse input argument for task " , task_name , "\n");
		kill(0, SIGTERM);
		return RT_GOMP_TASK_MANAGER_ARG_PARSE_ERROR;
	
	}
	
	start_time = {start_sec, start_nsec};
	end_time = {end_sec, end_nsec};
	
	char *barrier_name = argv[8];
	explicit_sync = std::stoi(std::string(argv[9])) == 1;
	int task_argc = argc - 10;                                             
	char **task_argv = &argv[10];

	//fetch the scheduler lock
	rescheduling_lock = smm::fetch<struct p_mutex>(reschedule_lock);

	//Wait at barrier for the other tasks but mainly to make sure scheduler has finished
	if ((ret_val = process_barrier::await_and_rearm_barrier("BAR_2")) != 0){

		print_module::print(std::cerr,  "ERROR: Barrier error for task " , task_name , "\n");
		kill(0, SIGTERM);
		return RT_GOMP_TASK_MANAGER_BARRIER_ERROR;

	}

	//Collect our first schedule and set it ups
	current_work = schedule.get_task(task_index)->get_current_work();
	current_period = schedule.get_task(task_index)->get_current_period();
	deadline = current_period;
	percentile = schedule.get_task(task_index)->get_percentage_workload();
	
	std::lock_guard<std::mutex> lk(con_mut);

	//print task information
	std::ostringstream task_info;
	std::string task_header = "|  Task " + std::to_string(task_index) + " (PID: " + std::to_string(getpid()) + ") |\n";
	size_t blank_size = task_header.size();

	print_module::buffered_print(task_info, "\n", std::string(blank_size - 1, '='), "\n");
	print_module::buffered_print(task_info, task_header);
	print_module::buffered_print(task_info, std::string(blank_size - 1, '='), "\n\n");

	//processor A info
	print_module::buffered_print(task_info, "Processor A Metrics: \n");
	print_module::buffered_print(task_info, "	- Lowest A Core: ", schedule.get_task(task_index)->get_current_lowest_processor_A(), "\n");
	print_module::buffered_print(task_info, "	- Current A Cores: ", schedule.get_task(task_index)->get_current_processors_A(), "\n");
	print_module::buffered_print(task_info, "	- Minimum A Cores: ", schedule.get_task(task_index)->get_min_processors_A(), "\n");
	print_module::buffered_print(task_info, "	- Maximum A Cores: ", schedule.get_task(task_index)->get_max_processors_A(), "\n");

	//processor B info
	print_module::buffered_print(task_info, "Processor B Metrics: \n");
	print_module::buffered_print(task_info, "	- Lowest B Core: ", schedule.get_task(task_index)->get_current_lowest_processor_B(), "\n");
	print_module::buffered_print(task_info, "	- Current B Cores: ", schedule.get_task(task_index)->get_current_processors_B(), "\n");
	print_module::buffered_print(task_info, "	- Minimum B Cores: ", schedule.get_task(task_index)->get_min_processors_B(), "\n");
	print_module::buffered_print(task_info, "	- Maximum B Cores: ", schedule.get_task(task_index)->get_max_processors_B(), "\n");

	//timing info
	print_module::buffered_print(task_info, "Timing Metrics: \n");
	print_module::buffered_print(task_info, "	- Period s: ", schedule.get_task(task_index)->get_current_period().tv_sec , " s\n");
	print_module::buffered_print(task_info, "	- Period ns: ", schedule.get_task(task_index)->get_current_period().tv_nsec , " ns\n\n");

	struct sched_param param;

	param.sched_priority = EXEC_PRIORITY;
	ret_val = sched_setscheduler(getpid(), SCHED_RR, &param);

	if (ret_val != 0)
 		print_module::print(std::cerr,  "WARNING: " , getpid() , " Could not set priority. Returned: " , errno , "  (" , strerror(errno) , ")\n");
	
	#ifndef OMP_OVERRIDE
		omp_set_num_threads(NUM_PROCESSOR_A);
	#endif

	#ifdef OMP_OVERRIDE

		omp(pragma_omp_parallel 
		{

			//set the affinity
			cpu_set_t local_cpuset;
			CPU_ZERO(&local_cpuset);
			CPU_SET(thread_id, &local_cpuset);

			pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &local_cpuset);

		});

	#else 

		#pragma omp parallel
		{

			//set the affinity
			cpu_set_t local_cpuset;
			CPU_ZERO(&local_cpuset);
			CPU_SET(omp_get_thread_num(), &local_cpuset);

			pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &local_cpuset);

		}

	#endif

	//Determine which threads are active and passive. Pin, etc.
	std::string active_cpu_string = "";
	std::string passive_cpu_string = "";

	//for CPUs that we MAY have at some point (practical max)
	for (int j = 1; j < NUM_PROCESSOR_A; j++){

		//Our first CPU is our permanent CPU 
		if (j == schedule.get_task(task_index)->get_current_lowest_processor_A()){
		
			cpu_set_t local_cpuset;
			CPU_ZERO(&local_cpuset);
			CPU_SET(j, &local_cpuset);

			//pin the main thread to this core
			pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &local_cpuset);
		
			schedule.get_task(task_index)->set_permanent_processor_A(j);

			#ifdef OMP_OVERRIDE
				omp.set_perm_cpu(j);
			#endif

		}
		if (j >= schedule.get_task(task_index)->get_current_lowest_processor_A() && j < schedule.get_task(task_index)->get_current_lowest_processor_A() + schedule.get_task(task_index)->get_current_processors_A()){

			global_param.sched_priority = EXEC_PRIORITY;

			active_cpu_string += std::to_string(j) + ", ";

			schedule.get_task(task_index)->push_back_processor_A(j);

		}

		else {

			global_param.sched_priority = SLEEP_PRIORITY;

			passive_cpu_string += std::to_string(j) + ", ";

		}
 
  	}

	//set the cpu mask
	processor_A_mask = schedule.get_task(task_index)->get_processor_A_mask();
	std::cout << "Process: " << task_index <<  " Initial Processor A Mask: " << (unsigned long long) processor_A_mask << std::endl;

	//print active vs passive CPUs
	print_module::buffered_print(task_info, "Processor A Core Configuration: \n");
	print_module::buffered_print(task_info, "	- Active: ", convertToRanges(active_cpu_string), "\n");
	//print_module::buffered_print(task_info, "	- Passive: ", convertToRanges(passive_cpu_string), "\n\n");

	//command line params
	print_module::buffered_print(task_info, "Command Line Parameters: \n");
	print_module::buffered_print(task_info, "	- Args: ", command_string, "\n\n");

	//flush all task info to terminal
	print_module::flush(std::cerr, task_info);

	//Initialize the program barrier
	bar.mc_bar_init(schedule.get_task(task_index)->get_current_processors_A());

	#ifdef PER_PERIOD_VERBOSE
		std::vector<uint64_t> period_timings;
	#endif

	//set the omp thread limit and mask
	omp_set_num_threads(schedule.get_task(task_index)->get_current_processors_A());

	#ifdef OMP_OVERRIDE
		omp.set_override_mask(processor_A_mask);
	#else
		set_active_threads(schedule.get_task(task_index)->get_processor_A_owned_by_process());
	#endif
		
	//update our gpu mask
	processor_B_mask = schedule.get_task(task_index)->get_processor_B_mask();

	//update our cpu C mask
	processor_C_mask = schedule.get_task(task_index)->get_processor_C_mask();

	//update our gpu D mask
	processor_D_mask = schedule.get_task(task_index)->get_processor_D_mask();

	// Initialize the task
	if (schedule.get_task(task_index) && task.init != NULL){

		if (task.init(task_argc, task_argv) != 0){

			print_module::print(std::cerr, "ERROR: Task initialization failed for task ", task_name, "\n");
			kill(0, SIGTERM);
			return RT_GOMP_TASK_MANAGER_INIT_TASK_ERROR;
		
		}

	}

	//call the core B update
	task.update_core_B(processor_B_mask);

	//call the core C update
	task.update_core_C(processor_C_mask);

	//call the core D update
	task.update_core_D(processor_D_mask);
	
	// Wait at barrier for the other tasks
	if ((ret_val = process_barrier::await_and_rearm_barrier(barrier_name)) != 0){

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

			//Fetch the mask of tasks which we are waiting on
			//(safe to call multiple times)
			schedule.get_task(task_index)->start_transition();

			//now fetch any resources which have been granted to us
			//(will return true when our mask is empty due to all
			//resources being granted)
			if (schedule.get_task(task_index)->get_processors_granted_from_other_tasks()){
				
				//semantically a bit strange, but we can only give or
				//take from each pool, so we call this and then call our
				//giving requirements

				//finalize the transition of resources to us (we now really own them)
				schedule.get_task(task_index)->acquire_all_processors();

				//transition any resources we are supposed to be giving up
				schedule.get_task(task_index)->give_processors_to_other_tasks();

				//call the reschedule procedure
				reschedule();

				//mark our transition as complete
				schedule.get_task(task_index)->set_mode_transition(true);
				needs_reschedule = false;

				//update our transition count (debugging)
				schedule.get_task(task_index)->set_num_adaptations(schedule.get_task(task_index)->get_num_adaptations() + 1);
				

			}

			else{

				//Gaining a processor that wasn't ready yet.
				print_module::print(std::cerr,  "Task ", getpid(), " can't reschedule yet due to resources not yet being available!\n");
			
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

    munmap(shared_array, 128 * sizeof(int));
    close(shm_fd);

	fflush(stdout);
	fflush(stderr);

return 0;
}	
