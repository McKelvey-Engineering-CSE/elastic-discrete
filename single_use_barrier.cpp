#include "single_use_barrier.h"
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

typedef struct
{
	unsigned value;
}
we;

static latch *get_barrier(const char *name, int *error_flag)
{
	int fd = shm_open(name, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
	if( fd == -1 )
	{
		std::perror("ERROR: single_use_barrier call to shm_open failed");
		*error_flag = RT_GOMP_SINGLE_USE_BARRIER_SHM_OPEN_FAILED_ERROR;
		return NULL;
	}

	int ret_val = ftruncate(fd, sizeof(latch));
	if( ret_val == -1 )
	{
		std::perror("ERROR: single_use_barrier call to ftruncate failed");
		*error_flag = RT_GOMP_SINGLE_USE_BARRIER_FTRUNCATE_FAILED_ERROR;
		return NULL;
	}

	latch *barrier = static_cast<latch*>(mmap(NULL, sizeof(latch), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));

	if (barrier == MAP_FAILED)
	{
		std::perror("ERROR: single_use_barrier call to mmap failed");
		*error_flag = RT_GOMP_SINGLE_USE_BARRIER_MMAP_FAILED_ERROR;
		return NULL;
	}
	
	ret_val = close(fd);
	if( ret_val == -1 )
	{
		std::perror("WARNING: single_use_barrier call to close file descriptor failed\n");
	}
	
	return barrier;
}

static void unmap_barrier( latch *barrier)
{
	int ret_val = munmap((void *) barrier, sizeof(latch));
	if (ret_val == -1)
	{
		std::perror("WARNING: single_use_barrier call to munmap failed\n");
	}
}

static void destroy_barrier(const char *name)
{
	int ret_val = shm_unlink(name);
	// If the name cannot be found, then the calling process lost the race 
	// to destroy the barrier which is not a problem. Report any other errors.
	if (ret_val == -1 && errno != ENOENT)
	{
		std::perror("WARNING: single_use_barrier call to shm_unlink failed\n");
	}
}

int init_single_use_barrier(const char *name, unsigned value)
{
	if (value == 0)
	{
		print(std::cerr, "ERROR: A barrier cannot be created for zero tasks\n");
		return RT_GOMP_SINGLE_USE_BARRIER_INVALID_VALUE_ERROR;
	}
	
	int error_flag = 0;
	latch *barrier = get_barrier(name, &error_flag);
	if (error_flag == 0)
	{
		barrier->init_latch(value);
		unmap_barrier(barrier);
	}
	
	return error_flag;
}

int await_single_use_barrier(const char *name)
{
	int error_flag = 0;
	 latch *barrier = get_barrier(name, &error_flag);
	if (error_flag == 0)
	{
		print(std::cout, "waiting at latch: ", name, "\nBarrier Value:", barrier->counter, "\n");

		barrier->arrive_and_wait();
		
		unmap_barrier(barrier);

		// Processes race to destroy the barrier. The race is semantically harmless.
		destroy_barrier(name);
	}
	
	return error_flag;
}
