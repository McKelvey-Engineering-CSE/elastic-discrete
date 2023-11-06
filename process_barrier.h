#ifndef RT_GOMP_SINGLE_USE_BARRIER_H
#define RT_GOMP_SINGLE_USE_BARRIER_H

#include "latch.h"

/*************************************************************************

single_use_barrier.h

This object This is functionally a latch. It is stored in shared memory
and allows the scheduler to ensure that all the threads from a given
task are ready when the task is first run. (As far as I can tell, this
latch is never used again)
        
        NOTE: an std::latch is likely to replace this in the future

**************************************************************************/

enum rt_gomp_single_use_barrier_error_codes
{
	RT_GOMP_SINGLE_USE_BARRIER_SUCCESS,
	RT_GOMP_SINGLE_USE_BARRIER_INVALID_VALUE_ERROR,
	RT_GOMP_SINGLE_USE_BARRIER_SHM_OPEN_FAILED_ERROR,
	RT_GOMP_SINGLE_USE_BARRIER_FTRUNCATE_FAILED_ERROR,
	RT_GOMP_SINGLE_USE_BARRIER_MMAP_FAILED_ERROR
};

class process_barrier : private latch {

	std::string name;
	std::mutex destruction_mux;

public:

	//inherit constructors
	using latch::latch; 

	//static function to destroy a barrier by force if needed
	static process_barrier* create_process_barrier(std::string barrier_name, int num_tasks);

	//static function to get a barrier object by name
	static process_barrier* get_process_barrier(std::string inname, int *error_flag);

	//static function to unmap a barrier
	static void unmap_process_barrier(process_barrier* barrier);

	//await function that allows latch functionality
	static int await_and_destroy_barrier(std::string barrier_name);
};

#endif /* RT_GOMP_SINGLE_USE_BARRIER_H */
