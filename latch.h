#ifndef LATCH_H
#define LATCH_H

#include <condition_variable>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <iostream>

#include "print.h"

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