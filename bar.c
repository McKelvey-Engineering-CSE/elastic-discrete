/* David Ferry - Sept 20, 2015
 *
 * This file implements a reenterable barrier designed for use with the
 * mixed criticality federated scheduling system.
 * */

/* NOTE! Sept 20, 2015 -
 * We're under a deadline constraint, so I've generally been very conservative
 * with the memory model semantics. It's likely that most of these constraints
 * could be relaxed while still preserving correctness, but that requires a
 * little more thought than I've got time for right now. */

#include "bar.h"

//We declare these here, they're used in futex.h. Normally libgomp would
//declare them extern and initialize them someplace else. I'm hoping that
//they're default initialized.
//int gomp_futex_wait, gomp_futex_wake;
#include <errno.h>
//#include <linux/futex.h>
//#include "futex.h"

#include <limits.h>

#include <sched.h> //For sched_getcpu
#include <stdio.h> //For printf
#include <unistd.h> //For sleep

//#define DAVID

//This value is used for each thread to keep track of which barrier invocation
//is the current barrier. We want hc threads to "drop through" barriers that
//have already been passed by the lc threads in lc mode.
__thread uint32_t thread_generation;


void mc_bar_init( mc_barrier *bar, uint32_t initial_total ){
	//printf("INIT BARRIER on CPU %d with INIT TOTAL %d\n", sched_getcpu(), initial_total);
	//fflush(stdout);	

	bar->generation     = 0;
	bar->generation2    = 0;
	bar->total_threads  = initial_total;
	bar->expected       = bar->total_threads;
	bar->num_hi_threads = 0;
	bar->is_switcher    = false;
	bar->locked         = false;
}


void mc_bar_reinit( mc_barrier *bar, uint32_t new_total ){
	//printf("INIT BARRIER on CPU %d with INIT TOTAL %d\n", sched_getcpu(), initial_total);
	//fflush(stdout);	

	//bar->generation     = 0;
	//bar->generation2    = 0;
	bar->total_threads  = new_total;
	bar->expected       = bar->total_threads;
	bar->num_hi_threads = 0;
	bar->is_switcher    = false;
	bar->locked	    = false;
}

//This function should be called when the system transitions to high crit mode.
//The parameter additional_hc_threads is the number of threads that join the
//computation after the mode switch.
//
//Note that this function should be called before any high criticality threads
//are released, otherwise a race condition exists and deadlock may result. 
void mc_bar_to_high_crit( mc_barrier *bar, uint32_t additional_hc_threads ){
	__atomic_store_n(&bar->num_hi_threads, additional_hc_threads, __ATOMIC_RELEASE);
}

//This function can only be called when it is guaranteed that no threads are
//in the barrier, and when it is guaranteed that high criticality threads won't
//try to enter the barrier in the future i.e. at the start or end of a period. 
//Only one thread should try to call this function.
void mc_bar_to_low_crit( mc_barrier *bar, uint32_t additional_hc_threads ){
	
	__atomic_store_n( &bar->locked , true, __ATOMIC_RELEASE);
	if( bar->expected != bar->total_threads ){
		printf("ERROR: Tried to update barrier during corrupted state! Exiting...\n");
		abort();
	}
	
	uint32_t current = bar->total_threads;
	uint32_t newcount = current - additional_hc_threads;
	
	__atomic_store_n(&bar->total_threads, newcount, __ATOMIC_RELEASE);

	__atomic_store_n( &bar->locked , false, __ATOMIC_RELEASE);
}

//This function spinwaits on the barrier value num_high_threads. The mode
//switcher thread will signal that it is finished by setting the value of
//bar->num_high_threads to zero.
void mc_spinwait( mc_barrier *bar ){
	uint32_t needs_switch;

	do {
		needs_switch = __atomic_load_n(&bar->num_hi_threads, __ATOMIC_ACQUIRE);
	} while( needs_switch != 0 );
}

//This function implements the switch protocol that updates the internal
//barrier.
void do_switch_protocol( mc_barrier *bar ){
//All threads will always first check whether the barrier needs to effect a
//mode switch. The variable bar->num_high_threads is used to signify this. If
//the value is positive, a mode switch is required. If the value is zero, no
//mode switch is required. If the value is negative, we are switching from
//high criticality to low criticality mode.

	uint32_t needs_switch = __atomic_load_n(&bar->num_hi_threads, __ATOMIC_ACQUIRE);
	//This clause only evaluates to true in the event of a mode switch, so
	//the builtin_expect tells the compiler to optimize for the case where
	//the expression needs_switch > 0 evaluates to false. 
	if( __builtin_expect(needs_switch != 0, 0) ) {
		//If needs_switch != 0, then a mode change is required. Some
		//thread will claim the role of mode switcher, and other
		//threads will spinwait until that thread has finished 
		//switching the barrier to high crit mode.

		bool cas_false = false;
		bool cas_true  = true;

		//First we have all threads try to grab the role of switcher,
		//which they do via an atomic compare and swap of the value
		//is_switcher. The thread that successfully switches the value
		//from false to true assumes the role of switcher.
		bool result = __atomic_compare_exchange( &bar->is_switcher,
                                                         &cas_false,
                                                         &cas_true,
                                                         false,
                                                         __ATOMIC_ACQ_REL,
                                                         __ATOMIC_ACQUIRE); 

		if( result ){
			//The thread that gets here has assumed the role of
			//switcher. The do-while loop will break whent he 
			//switcher successfully breaks updates the number
			//of expected threads.
			uint32_t current = 0, newval = 0, new_expected = 0;
			do{
				//If the expected number of threads is zero,
				//then some thread is currently waking up
				//sleeping threads and resetting the barrier
				//state, so we wait until expected > 0. 
				while ( __atomic_load_n(&bar->expected, __ATOMIC_ACQUIRE) == 0 ) {};
				current = __atomic_load_n(&bar->expected, __ATOMIC_ACQUIRE);
				//needs_switch contains num of extra threads
				newval = current + needs_switch;
				//bool res2 = __atomic_compare_exchange(&bar->expected,
                                //                                      &current, 
                                //                                      &newval, 
                                //                                      false,
                                //                                      __ATOMIC_ACQ_REL,
                                //                                      __ATOMIC_ACQUIRE);

				new_expected = __atomic_load_n(&bar->expected, __ATOMIC_ACQUIRE);
			} while ( new_expected != newval );

			//Updating bar->total here is safe because at this
			//point the barrier is currently expecting the switcher
			//thread (this thread) before it resets.
			__atomic_add_fetch(&bar->total_threads, needs_switch, __ATOMIC_ACQ_REL);
			
			//We are done updating the barrier internal state, and
			//we set bar->num_hi_threads to zero to signal to any
			//threads busy wating in the else clause.
			__atomic_store_n(&bar->num_hi_threads, 0, __ATOMIC_RELEASE);

		} else {
			//Else we did not acquire the role of is_switcher and
			//we will spinwait here until the switcher thread is
			//finished.
		
			mc_spinwait( bar );	
		}
	} 
}

//We use the GCC OpenMP implementation of futex sleeping for convinience.
//The futex allows a thread to sleep on the value located at a specific region
//of memory. In our case, the threads sleep on the current value of
//bar->generation. When a thread is put to sleep, it first checks atomically
//that bar->generation == current_generation, and if so it will sleep. When
//it wakes up, it verifies that bar->generation has changed. 
void mc_bar_wake_up_threads(mc_barrier *bar){
	
	#ifdef USE_FUTEX	
	futex_wake ((int *) &bar->generation, INT_MAX);
	#endif //USE_FUTEX

	#ifdef USE_C11_CV
	//Retry the notify_all() untill all threads have successfully left
	//the barrier. Each thread will increment bar->expected as it leaves.
	do {
	bar->bar_cv.notify_all();
	} while ( __atomic_load_n(&bar->expected, __ATOMIC_ACQUIRE) != bar->total_threads );
	#endif //USE_C11_CV

}

void mc_bar_put_self_to_sleep( mc_barrier *bar, uint32_t current_gen){
	//Note that this implementation allows racy behavior beteween the
	//waking thread and the sleeping threads. After the waking thread
	//increments bar->generation, a thread entering this function will
	//no longer actually futex sleep. 

	#ifdef USE_FUTEX
	do{
		futex_wait ((int *) &bar->generation, current_gen );
	} while (__atomic_load_n(&bar->generation, __ATOMIC_ACQUIRE) == current_gen);
	#endif //USE_FUTEX

	#ifdef USE_C11_CV
	std::unique_lock<std::mutex> lock (bar->bar_m);
	bar->bar_cv.wait( lock, [current_gen,bar]{ return !(__atomic_load_n(&bar->generation, __ATOMIC_ACQUIRE) == current_gen );} );
	#endif //USE_C11_CV

}

void mc_bar_wait( mc_barrier *bar ){

	//Before we do anything, check to see if the barrier is locked.
	while( __atomic_load_n(&bar->locked, __ATOMIC_ACQUIRE) == true )
	{ /* busy wait */ }

	do_switch_protocol( bar );

	//From this point on, we do a regular barrier_wait

	//We use the generation variable to indicate how many times this 
	//barrier has been reset. This is so that, when a new thread starts
	//to participate in the computation, it can drop through previous
	//calls to mc_bar_wait
	uint32_t current_generation = __atomic_load_n(&bar->generation, __ATOMIC_ACQUIRE);

	//We use two generation variables to make sure that all threads have left
	//a barrier from a previous geneartion before new threads can begin to wait on it.
	//Otherwise a thread could speed through and go to sleep on the barrier while other
	//threads are waking up, and thereby skip the barrier.
	while( __atomic_load_n(&bar->generation2, __ATOMIC_ACQUIRE) < current_generation )
	{ /*busy wait*/ }

		#ifdef DAVID
			printf("THREAD GEN: %d     CURR GEN: %d\n", thread_generation, current_generation);
		#endif 

	if( thread_generation == current_generation){
		uint32_t result = __atomic_add_fetch( &bar->expected, -1, __ATOMIC_ACQ_REL );

		#ifdef DAVID
			printf("RESULT OF SUBTRACTING AND FETCHING EXPECTED %d\n", result);
		#endif 

		if( result == 0 ) {
			//We are the last thread through the barrier, so we
			//wake up all other threads.
			__atomic_add_fetch( &bar->generation, 1, __ATOMIC_ACQ_REL );

			__atomic_add_fetch( &bar->expected, 1, __ATOMIC_ACQ_REL );

			mc_bar_wake_up_threads(bar);

			//We use a second generation value to require that all threads leave
			//the barrier before any new threads are allowed in.
			__atomic_add_fetch( &bar->generation2, 1, __ATOMIC_ACQ_REL );

		} else {
			//We are not the last thread through the barrier,
			//so we put ourselves to sleep. When we leave, in increment the expected
			//count so we can signal to the waking thread that everything has passed the
			//barrier successfully.
			mc_bar_put_self_to_sleep(bar, current_generation);
			__atomic_add_fetch( &bar->expected, 1, __ATOMIC_ACQ_REL );
		}
	}

	//Update how many bar_wait calls we've encountered
	thread_generation++;
}
