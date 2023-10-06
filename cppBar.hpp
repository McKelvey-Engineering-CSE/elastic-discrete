#include <barrier>
#include <latch>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <barrier>

#ifndef CPPBARRIER_HPP
#define CPPBARRIER_HPP

class cppBarrier {

//Tyler Martin 
//October 5 2023
//This class is a cpp_20 compliant
//barrier implementation to replace the C 
//variant
private:
	uint32_t generation __attribute__(( aligned(64) ));
	uint32_t generation2;
	uint32_t total_threads; 
	uint32_t expected __attribute__(( aligned(64) ));
	uint32_t num_hi_threads; //This value serves as data storage
                                 //as well as a lock on the barrier
	bool is_switcher; //Used to synchronize during mode switch
	bool locked;
	
    std::mutex bar_m;
	std::condition_variable bar_cv;

    //Cpp_20 added vars
    //std::barrier cppBarrier;
    std::mutex writeMux;
    std::mutex locked_to_low;
    std::shared_mutex switcher_lock;

    //functions that don't need public access
    void mc_spinwait();

public:

    //NOTE: constructor functionally serves as init handler, which means that each barrier must be created with initial thread value passed to it
    cppBarrier( uint32_t total_threads, uint32_t initial_total ):   generation(0), 
                                                                    generation2(0), 
                                                                    total_threads(initial_total), 
                                                                    expected(total_threads), 
                                                                    num_hi_threads(0), 
                                                                    is_switcher(false), 
                                                                    locked(false){}

    //This initializes the barrier for operation with initial_total number of
    //threads. Normally this will be the number of low criticality threads.
    void mc_bar_init(uint32_t initial_total);

    //retaining old functions in-case any other calls to this were made
    void mc_bar_reinit(uint32_t new_total);

    //This notifies the barrier that it should now expect an additional number
    //of threads to participate. This should normally be called when the high
    //criticality system transistion happens. Note that this method requires
    //the *additional* number of threads, not the *total* number of high
    //criticality threads.
    void mc_bar_to_high_crit(uint32_t additional_hc_threads);

    //This notifies the barrier that it should expect fewer threads to participate.
    //This function is only called after a transition to high criticality, i.e. not
    //at system initialization. 
    void mc_bar_to_low_crit(uint32_t additional_hc_threads);

    //Threads should call this to wait at the barrier
    void mc_bar_wait();

    //TODO: figure out if this should be public callable
    void do_switch_protocol();

};
#endif