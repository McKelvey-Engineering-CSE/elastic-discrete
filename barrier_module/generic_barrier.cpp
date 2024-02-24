#include "generic_barrier.h"
#include <iostream>

void generic_barrier::init(int in){
    std::lock_guard<std::mutex> lock(mut);
    count = in;
}

void generic_barrier::arrive_and_wait()
{
    //lock mutex
    std::unique_lock<std::mutex> lock(mut);

    print_module::print(std::cerr, "Barrier Count: ", count, " | ", getppid(), "\n");

    //check if exit or wait
    if (--count == 0) {
        lock.unlock();
        cv.notify_all();
    } 
    
    else {
        lock.unlock();
        while(count != 0){
            cv.wait(r_mutex);
        }
    }

    //handle return function
    return_function();
}

void generic_barrier::return_function(){

    if (execute_function){

        print_module::print(std::cerr, "running exit function\n");

        //if "scheduler" is the only one doing it
        if (scheduler_only){
            return;
        }

        else{
            ret_function();
        }
    }
}