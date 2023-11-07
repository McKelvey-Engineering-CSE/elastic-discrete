#ifndef _thread_barrier_H
#define _thread_barrier_H

/*************************************************************************

thread_barrier.h

This object This is a reenterable barrier that is designed for use with the
discrete elastic scheduling system. It provides a single class
called thread_barrier that all threads share, which contins a function, 
mc_bar_to_high_crit that is called when the system transitions to high 
criticality mode and releases extra high-criticality threads.

The primary problem this barrier solves is that the internal state of the
barrier can be changed at any time when high criticality mode is entered.


Class : thread_barrier (inherits class latch)

		This class implements the barrier in a c++ idiomatic way.
        In the future, this class will implement atomic structures
        and calls using locks and mutexes to synchronize processes.

**************************************************************************/

#include <stdint.h>
#include <stdbool.h>
#include <condition_variable>
#include <mutex>

#include "print.h"
#include "latch.h"

//#define USE_FUTEX
#define USE_C11_CV
#if defined(USE_FUTEX) && defined(USE_C11_CV)
#error "Cannot define both USE_FUTEX and USE_C11_CV!"
#endif

class thread_barrier : latch {

    //grab all inherited constructors
    using latch::latch; 

	//The aligned attribute ensures that this data field occupies a single
	//cache line. The normal layout will put generation and total_threads
	//together on one line, and expected on another line.
	//
	//We want to make sure that expected is on a separate line from
	//generation and total_threads, since the former is frequently written
	//and the latter are not, which would cause excessive cache
	//invalidations.
	uint32_t generation __attribute__(( aligned(64) ));
	uint32_t generation2;
	uint32_t total_threads; 
	uint32_t expected __attribute__(( aligned(64) ));
	uint32_t num_hi_threads; //This value serves as data storage
                                 //as well as a lock on the barrier
	bool is_switcher; //Used to synchronize during mode switch
	bool locked;

	#ifdef USE_C11_CV
	std::mutex bar_m;
	std::condition_variable bar_cv;
	#endif

    //functions who are called by already active threads
    void do_switch_protocol();
    void mc_spinwait();

public:

    //simple constructors for future use
    thread_barrier():   generation(0), 
                generation2(0), 
                total_threads(0), 
                expected(0), 
                num_hi_threads(0), 
                is_switcher(false), 
                locked(false){}

    thread_barrier( uint32_t total_threads, uint32_t initial_total ):   generation(0), 
                                                            generation2(0), 
                                                            total_threads(initial_total), 
                                                            expected(total_threads), 
                                                            num_hi_threads(0), 
                                                            is_switcher(false), 
                                                            locked(false){}

    //This initializes the barrier for operation with initial_total number of
    //threads. Normally this will be the number of low criticality threads.
    void mc_bar_init(uint32_t initial_total);

    //This resets the number of expected threads, not fully tested... use at own risk
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

};

#endif //_thread_barrier_H

