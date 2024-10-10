#include "process_barrier.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <cstdlib>
#include <errno.h>
#include <time.h>

#include <iostream>
#include <cerrno>

#include "generic_barrier.h"
#include "memory_allocator.h"

#ifdef LOG_STATES

#include "print_module.h"

#endif

//static function to create a new process_barrier
process_barrier* process_barrier::create_process_barrier(std::string barrier_name, int num_tasks, std::function<void()> infunction, bool inall, bool inexecution){

	int ret_val = 0;
	process_barrier* barrier = get_process_barrier(barrier_name, &ret_val, infunction, inall, inexecution, true);

	if (ret_val == -1){
		std::perror("FAILURE: creation of a process barrier has failed. Program will now exit\n");
		return nullptr;
	}

	barrier->init(num_tasks);
	return (barrier);

}

//static function to grab a barrier from shared memory based on the name
process_barrier* process_barrier::get_process_barrier(std::string inname, int *error_flag, std::function<void()> infunction, bool inall, bool inexecution, bool create)
{

	//call the memory allocator
	process_barrier *barrier;

	if (create)
		barrier = shared_memory_module::allocate<process_barrier>(inname, size_t(1), infunction, inall, inexecution);

	else
		barrier = shared_memory_module::fetch<process_barrier>(inname);

	if (barrier == nullptr){
		exit(-1);
	}
	
	return barrier;
}

//static function to unmap a shared memory process_barrier from a process
void process_barrier::unmap_process_barrier(process_barrier *barrier)
{
	int ret_val = munmap((void *) barrier, sizeof(process_barrier));
	if (ret_val == -1)
	{
		std::perror("WARNING: process_barrier call to munmap failed\n");
	}
}

//static function that allows waiting at the process_barrier
int process_barrier::await_and_destroy_barrier(std::string barrier_name)
{

	#ifdef LOG_STATES 
		print_module::print(std::cout, getpid(), " | waiting at barrier: ", barrier_name, "\n");
	#endif

	int ret_val=0;
	process_barrier* barrier = get_process_barrier(barrier_name, &ret_val);

	if (ret_val != 0)
		return ret_val;
	barrier->arrive_and_wait();

	shared_memory_module::detatch<process_barrier>(barrier);
	shared_memory_module::delete_memory<process_barrier>(barrier_name);

	return 0;
}

//static function that allows waiting at the process_barrier
int process_barrier::await_and_rearm_barrier(std::string barrier_name)
{
	#ifdef LOG_STATES 
		print_module::print(std::cout, getpid(), " | waiting at barrier: ", barrier_name, "\n");
	#endif

	int ret_val=0;
	process_barrier* barrier = get_process_barrier(barrier_name, &ret_val);

	if (ret_val != 0)
		return ret_val;
	barrier->arrive_and_wait(true);

	return 0;
}

//static function that allows for barrier destruction
int process_barrier::destroy_barrier(std::string barrier_name){
	return shared_memory_module::delete_memory<process_barrier>(barrier_name);
}