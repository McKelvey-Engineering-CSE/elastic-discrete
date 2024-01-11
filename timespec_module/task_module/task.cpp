#include "task.h"
#include <sched.h>
#include <omp.h>
#include "futexold.h"
#include <limits.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>

#define gettid() syscall(SYS_gettid)
//#define DAVID

inline void futex_sleep(){
	old_futex_wait(&futex_val, 0);
}

inline void futex_wakeup(){
	old_futex_wake(&futex_val, INT_MAX);
}

void mode_change_setup() {
  #ifdef TRACING
  fprintf( fd, "thread %ld: starting mode_change_setup\n", gettid());
  fflush( fd );
   #endif

		if(!(active[omp_get_thread_num()]))
		{
	
			#ifdef TRACING
			fprintf( fd, "thread %ld: putting self to sleep\n", gettid());
			fflush( fd );
			#endif
		
			futex_sleep();
			
			#ifdef TRACING
			fprintf( fd, "thread %ld: woke up!\n", gettid());
			fflush( fd );
			#endif	

		}
	return;
}

void mode_change_finish(){
	#ifdef TRACING
	fprintf( fd, "thread %ld: starting mode_change_finish\n", gettid());
	fflush( fd );
	#endif

	//Decrement the counter
	__atomic_add_fetch( &total_remain, -1, __ATOMIC_ACQ_REL);

	#ifdef DAVID
	double start, end;
	start = omp_get_wtime();
	#endif

	//Notify waiting high crit threads
	while( __atomic_load_n( &total_remain, __ATOMIC_ACQUIRE ) > 0 )
	{
		futex_wakeup();
	}  
	#ifdef DAVID
	end = omp_get_wtime();
	printf("Thread %d spent %0.1fus in busy wait\n", cpus[omp_get_thread_num()], (end-start)*1000000.0);
	#endif

}

