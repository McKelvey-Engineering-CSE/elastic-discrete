#include "cppBar.hpp"
//retaining for code interopability; may be removed later if deemed not needed
//in a rewrite
void cppBarrier::mc_bar_reinit(uint32_t new_total){
	total_threads =new_total; 
	expected = total_threads;
	num_hi_threads = 0;
	is_switcher = false; 
	locked = false;
}

//This function should be called when the system transitions to high crit mode.
//The parameter additional_hc_threads is the number of threads that join the
//computation after the mode switch.
//
//Note that this function should be called before any high criticality threads
//are released, otherwise a race condition exists and deadlock may result. 
void cppBarrier::mc_bar_to_high_crit(uint32_t additional_hc_threads ){

    const std::lock_guard<std::mutex> lock(writeMux);
    num_hi_threads = additional_hc_threads;

}

//This function can only be called when it is guaranteed that no threads are
//in the barrier, and when it is guaranteed that high criticality threads won't
//try to enter the barrier in the future i.e. at the start or end of a period. 
//Only one thread should try to call this function.
void cppBarrier::mc_bar_to_low_crit(uint32_t additional_hc_threads ){
	
    const std::lock_guard<std::mutex> lock(locked_to_low);

    //leave for now, update later if needed
	if(expected != total_threads ){
		std::cout << "ERROR: Tried to update barrier during corrupted state! Exiting...\n";
		abort();
	}
	
    total_threads = total_threads - additional_hc_threads;
}

//This function spinwaits on the barrier value num_high_threads. The mode
//switcher thread will signal that it is finished by setting the value of
//bar->num_high_threads to zero.
void cppBarrier::mc_spinwait(){
	uint32_t needs_switch;

	do {
		needs_switch = __atomic_load_n(&num_hi_threads, __ATOMIC_ACQUIRE);
	} while( needs_switch != 0 );
}

//This function implements the switch protocol that updates the internal
//barrier.
void cppBarrier::do_switch_protocol(){
//All threads will always first check whether the barrier needs to effect a
//mode switch. The variable bar->num_high_threads is used to signify this. If
//the value is positive, a mode switch is required. If the value is zero, no
//mode switch is required. If the value is negative, we are switching from
//high criticality to low criticality mode.

	uint32_t needs_switch = __atomic_load_n(&num_hi_threads, __ATOMIC_ACQUIRE);
	//This clause only evaluates to true in the event of a mode switch, so
	//the builtin_expect tells the compiler to optimize for the case where
	//the expression needs_switch > 0 evaluates to false. 
	if(__builtin_expect(needs_switch != 0, 0) ) {
		//If needs_switch != 0, then a mode change is required. Some
		//thread will claim the role of mode switcher, and other
		//threads will spinwait until that thread has finished 
		//switching the barrier to high crit mode.

		//bool cas_false = false;
		//bool cas_true  = true;

		//First we have all threads try to grab the role of switcher,
		//which they do via an atomic compare and swap of the value
		//is_switcher. The thread that successfully switches the value
		//from false to true assumes the role of switcher.

		//Tyler 10-5-2023
		//replaced the atomic grab and switch with a try lock system
		//not sure how this will behave when end of scope is reached
		if(switcher_lock.try_lock()){
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

				//not sure there is a better way to do this, so leaving it for now - Tyler
				while ( __atomic_load_n(&expected, __ATOMIC_ACQUIRE) == 0 ) {};
				current = __atomic_load_n(&expected, __ATOMIC_ACQUIRE);

				//needs_switch contains num of extra threads
				newval = current + needs_switch;
				new_expected = __atomic_load_n(&expected, __ATOMIC_ACQUIRE);
			
			} while ( new_expected != newval );

			//Updating bar->total here is safe because at this
			//point the barrier is currently expecting the switcher
			//thread (this thread) before it resets.

			//need to check the behavior of atomic functions and std concurrency - Tyler
			__atomic_add_fetch(&total_threads, needs_switch, __ATOMIC_ACQ_REL);
			
			//We are done updating the barrier internal state, and
			//we set bar->num_hi_threads to zero to signal to any
			//threads busy wating in the else clause.

			//need to unlock the shared mutex as well for any further switches - Tyler
			__atomic_store_n(&num_hi_threads, 0, __ATOMIC_RELEASE);
			switcher_lock.unlock();

		} else {
			//Else we did not acquire the role of is_switcher and
			//we will spinwait here until the switcher thread is
			//finished.
		
			mc_spinwait();	
		}
	} 
}