#include "generic_barrier.h"
#include <iostream>

//#define LOG_STATES

void generic_barrier::init(int in){
    std::lock_guard<std::mutex> lock(mut);
    count = in;
}

void generic_barrier::arrive_and_wait()
{
    //lock mutex
    #ifdef LOG_STATES 
        std::cout << "waiting at lock" << std::endl;
    #endif 

    r_mutex.lock();

    #ifdef LOG_STATES 
        std::cout << "Lock Grabbed" << std::endl; 
    #endif

    #ifdef LOG_STATES 
        print_module::print(std::cerr, "Barrier Count: ", count, " | ", getppid(), "\n");
    #endif

    //check if exit or wait
    if (--count == 0) {

        #ifdef LOG_STATES 
            std::cout << "Notifying All Waiting" << std::endl;
        #endif 

        r_mutex.unlock();
        cv.notify_all();
    } 
    
    else {
        r_mutex.unlock();
        while(count != 0){

            #ifdef LOG_STATES 
                std::cout << "Checking before wait..." << std::endl;
            #endif

            cv.wait(r_mutex);
        }

        #ifdef LOG_STATES 
            std::cout << "out of wait" << std::endl;
        #endif
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