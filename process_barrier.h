#ifndef RT_GOMP_SINGLE_USE_BARRIER_H
#define RT_GOMP_SINGLE_USE_BARRIER_H

#include "latch.h"

/*************************************************************************

process_barrier.h

This object is an extension of the latch object defined in latch.h
It serves as a reinitializable barrier that can be used to synchronize 
processes. It has multiple static methods which allow an unusual accessing
pattern, as the constructor itself is only ever called by the object
when it creates its own memory segment. 
        
objects:
	enum: rt_gomp_single_use_barrier_error_codes
		simple error codes for various functions

	class: process_barrier (inheriting from latch)
		barrier for interprocess synchronization

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
	static process_barrier* create_process_barrier(std::string barrier_name, int num_tasks, std::function<void()> infunction = nullptr, bool inall = false, bool inexecution = false);

	//static function to get a barrier object by name
	static process_barrier* get_process_barrier(std::string inname, int *error_flag, std::function<void()> infunction = nullptr, bool inall = false, bool inexecution = false);

	//static function to unmap a barrier
	static void unmap_process_barrier(process_barrier* barrier);

	//await function that allows latch functionality
	static int await_and_destroy_barrier(std::string barrier_name);
};

#endif /* RT_GOMP_SINGLE_USE_BARRIER_H */
