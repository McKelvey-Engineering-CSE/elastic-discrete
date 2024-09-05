#include "generic_barrier.h"
#include <iostream>
#include <unistd.h> 

#ifdef LOG_STATES

#include "print_module.h"

#endif

void generic_barrier::init(int in){
    r_mutex.lock();
    count = in;
    initial_count = count;
    r_mutex.unlock();
}

void generic_barrier::arrive_and_wait(bool rearm)
{
    //lock mutex
    #ifdef LOG_STATES 
        print_module::print(std::cout, getpid(), " | waiting at lock\n");
    #endif 

    r_mutex.lock();

    #ifdef LOG_STATES 
        print_module::print(std::cout, getpid(), " | Lock Grabbed\n");
    #endif

    //check if exit or wait
    if (--count == 0) {

        #ifdef LOG_STATES 
            print_module::print(std::cout, getpid(), " | Notifying All Waiting\n");
        #endif 

        if (rearm)
            count = initial_count;

        generation += 1;

        r_mutex.notify_all();
        r_mutex.unlock();
    } 
    
    else {
        int my_generation = generation;

        while(count != 0 && generation == my_generation){

            #ifdef LOG_STATES 
                print_module::print(std::cout, getpid(), " | Checking before wait...\n");
            #endif

            r_mutex.wait();

            #ifdef LOG_STATES 
                print_module::print(std::cout, getpid(), " | got notified..\n");
            #endif
        }

        #ifdef LOG_STATES 
            print_module::print(std::cout, getpid(), " | out of wait\n");
        #endif
    }

}

void generic_barrier::return_function(){

    if (execute_function){

        //if "scheduler" is the only one doing it
        if (scheduler_only){
            return;
        }

        else{
            ret_function();
        }
    }
}