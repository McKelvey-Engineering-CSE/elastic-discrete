#include "generic_barrier.h"

void generic_barrier::init(int in){
    std::lock_guard<std::mutex> lock(mut);
    count = in;
}

void generic_barrier::arrive_and_wait()
{
    //lock mutex
    std::unique_lock<std::mutex> lock(mut);

    //check if exit or wait
    if (--count == 0) {
        cv.notify_all();
    } 
    
    else {
        cv.wait(lock, [this] { return count == 0;});
    }

    //handle return function
    return_function();
}

void generic_barrier::return_function(){

    if (execute_function){

        print(std::cerr, "running exit function\n");

        //if "scheduler" is the only one doing it
        if (scheduler_only){
            return;
        }

        else{
            ret_function();
        }
    }
}