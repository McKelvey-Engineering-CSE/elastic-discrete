#ifndef LATCH_H
#define LATCH_H

#include <condition_variable>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <iostream>

#include "print.h"

class latch {

public:
    std::atomic<int> counter;
    std::condition_variable cv_;
    std::mutex mut_;
    std::shared_mutex mut_update_;

public:

    //Allow construction of a latch without a known
    //size to begin with.
    latch(int value = 1) : counter(value) {}

    //reset our value (I know this is actually a barrier now, but whatever)
    void init_latch(int value);

    //decrement counter
    void count_down();

    //just returns whether or not the barrier is set
    bool try_wait();

    //stall at barrier without dec
    void wait();

    //arrive at barrier, dec and wait
    void arrive_and_wait();
};

#endif