#ifndef LATCH_H
#define LATCH_H

#include <condition_variable>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <iostream>
#include <functional>
#include <pthread.h>

#include "process_primitives.h"

/*************************************************************************

generic_barrier.h

Custom barrier object. Functions identically to C++20 barriers, but can be 
stored in shared memory, reinitialized to a new count, and does not require 
a return type function to be constructed. If provided with a function type
and a pointer to a function, that function will be executed when the processes 
exit the barrier. The barrier function exit strategy can either operate on a 
scheduler mode, where only the scheduling process/thread executes the exit
function, or on a per-thread basis where each process/thread executes the 
function as it leaves.

Objects : generic_barrier <template>

**************************************************************************/

class generic_barrier 
{

protected:
    std::mutex mut;
    pthread_mutex_t& mut_handle = *(pthread_mutex_t*)mut.native_handle();
    p_mutex r_mutex;
    std::size_t count;
    std::size_t initial_count;
    std::function<void()> ret_function;
    bool scheduler_only = true;
    bool execute_function = false;
    int generation = 0;

public:

    //construct generic_barrier
    explicit generic_barrier() : count(1) {  }
    
    explicit generic_barrier(std::size_t incount) : count(incount) {  }

    explicit generic_barrier(std::size_t incount, std::function<void()> inret_func) :   count(incount), 
                                                                                        ret_function(inret_func), 
                                                                                        execute_function(true) {  }

    explicit generic_barrier(std::size_t incount, std::function<void()> inret_func, bool all) : count(incount), 
                                                                                                ret_function(inret_func), 
                                                                                                scheduler_only(all), 
                                                                                                execute_function(true) {  }

    explicit generic_barrier(std::size_t incount, std::function<void()> inret_func, bool all, bool execute) :   count(incount), 
                                                                                                                ret_function(inret_func), 
                                                                                                                scheduler_only(all), 
                                                                                                                execute_function(execute) {  }

    //reinit for barrier mode
	void init(int in);

    //arrive at the barrier and wait
    void arrive_and_wait(bool rearm = false);

    //execute return function
    void return_function();
};


#endif