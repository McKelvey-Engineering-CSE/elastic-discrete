#ifndef LATCH_H
#define LATCH_H

#include <condition_variable>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <iostream>

#include "print.h"

/*************************************************************************

latch.h

Custom barrier object. Functions identically to C++20 barriers, but can be 
stored in shared memory, reinitialized to a new count, and does not require 
a return type function to be constructed. If provided with a function type
and a pointer to a function, that function will be executed when the processes 
exit the barrier. The barrier function exit strategy can either operate on a 
scheduler mode, where only the scheduling process/thread executes the exit
function, or on a per-thread basis where each process/thread executes the 
function as it leaves.

Objects : latch

**************************************************************************/

class latch 
{

std::mutex mut;
std::condition_variable cv;
std::size_t count;

public:

    //construct latch
    explicit latch(std::size_t incount) : count(incount) { }

    //reinit for barrier mode
	void init_latch(int in);

    //arrive at the barrier and wait
    void arrive_and_wait();
};


#endif