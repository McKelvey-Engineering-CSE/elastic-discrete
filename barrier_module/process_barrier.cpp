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

#include "print_module.h"
#include "generic_barrier.h"
#include "memory_allocator.h"


//static function to create a new process_barrier
process_barrier* process_barrier::create_process_barrier(std::string barrier_name, int num_tasks, std::function<void()> infunction, bool inall, bool inexecution){

	int ret_val = 0;
	process_barrier* barrier = get_process_barrier(barrier_name.c_str(), &ret_val, infunction, inall, inexecution);

	if (ret_val == -1){
		std::perror("FAILURE: creation of a process barrier has failed. Program will now exit\n");
		return nullptr;
	}

	barrier->init(num_tasks+1);
	return (barrier);

}

//static function to grab a barrier from shared memory based on the name
process_barrier* process_barrier::get_process_barrier(std::string inname, int *error_flag, std::function<void()> infunction, bool inall, bool inexecution)
{

	//call the memory allocator
	process_barrier *barrier = shared_memory_module::allocate<process_barrier>(inname, size_t(1), infunction, inall, inexecution);

	if (barrier == nullptr){
		print_module::print(std::cerr, "Cannot continue, a print buffer could not be allocated\n");
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

	int ret_val=0;
	process_barrier* barrier = get_process_barrier(barrier_name, &ret_val);

	if (ret_val != 0)
		return ret_val;

	barrier->arrive_and_wait();
	
	unmap_process_barrier(barrier);

	ret_val = shm_unlink(barrier_name.c_str());

	if (ret_val == -1 && errno != ENOENT)
	{
		std::perror("WARNING: process_barrier call to shm_unlink failed\n");
	}

	return 0;
}
