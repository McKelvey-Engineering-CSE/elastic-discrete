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

#include "print.h"
#include "latch.h"


//static function to create a new process_barrier
process_barrier* process_barrier::create_process_barrier(std::string barrier_name, int num_tasks){

	int ret_val = 0;
	process_barrier* barrier = get_process_barrier(barrier_name.c_str(), &ret_val);

	if (ret_val == -1){
		std::perror("FAILURE: creation of a process barrier has failed. Program will now exit\n");
		return nullptr;
	}

	barrier->init(num_tasks+1);
	return (barrier);

}

//static function to grab a barrier from shared memory based on the name
process_barrier* process_barrier::get_process_barrier(std::string inname, int *error_flag)
{

	int fd = shm_open(inname.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
	if( fd == -1 )
	{
		std::perror("ERROR: process_barrier call to shm_open failed");
		*error_flag = RT_GOMP_SINGLE_USE_BARRIER_SHM_OPEN_FAILED_ERROR;
		return nullptr;
	}

	int ret_val = ftruncate(fd, sizeof(process_barrier));
	if( ret_val == -1 )
	{
		std::perror("ERROR: process_barrier call to ftruncate failed");
		*error_flag = RT_GOMP_SINGLE_USE_BARRIER_FTRUNCATE_FAILED_ERROR;
		return nullptr;
	}

	process_barrier *barrier = new ((mmap(NULL, sizeof(process_barrier), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, fd, 0))) process_barrier(1);

	if (barrier == MAP_FAILED)
	{
		std::perror("ERROR: process_barrier call to mmap failed");
		*error_flag = RT_GOMP_SINGLE_USE_BARRIER_MMAP_FAILED_ERROR;
		return nullptr;
	}
	
	ret_val = close(fd);
	if( ret_val == -1 )
	{
		std::perror("WARNING: process_barrier call to close file descriptor failed\n");
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
