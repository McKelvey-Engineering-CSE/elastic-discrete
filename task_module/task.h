#ifndef RT_GOMP_TASK_H
#define RT_GOMP_TASK_H

/*************************************************************************

task.h

This object is what the presented custom task is compiled against. It
provides an interface for the task manager to interact with a given task
throug the use of he task_t struct, which defines each of the functions
that are needed by a custom task to be utilized here.


Struct : task_t

		Task struct type used by task_manager.cpp to control a task.
        
**************************************************************************/

#include <condition_variable>
#include <mutex>
#include <atomic>
#include "thread_barrier.h"
#include <sched.h>
#include "include.h"

//#define TRACING

#ifdef TRACING
extern FILE * fd;
#endif

#ifdef OMP_OVERRIDE

	#include "omp_replacement.hpp"

	extern OMPThreadPool* omp_threadpool;

#endif

// Task struct type used by task_manager.cpp to control a task.
typedef struct
{
	int (*init)(int argc, char *argv[]);
	int (*run)(int argc, char *argv[]);
	int (*finalize)(int argc, char *argv[]);
	void (*update_core_B)(__uint128_t mask);
	void (*update_core_C)(__uint128_t mask);
	void (*update_core_D)(__uint128_t mask);
}
task_t;

extern bool active_threads[64];
extern int current_mode;
extern double percentile;

extern void modify_self(timespec new_value);

extern int get_current_mode();

extern void set_cooperative(bool value);

extern void mimic_simulator(int task_index);

extern void set_victim_prevention(bool value);

extern int get_previous_mode();

// Used to determine current task and its features.
extern bool missed_dl;

// Task struct that should be defined by the real time task.
extern task_t task;

extern timespec current_period;
extern timespec current_work;

extern bool mode_change_finished;

// We need a special barrier that understands the mixed-criticality mode
// transition and the fact that different numbers of threads are expected at
// the barrier at different times. This provides that, and should be used
// instead of any other barrier. Initialized in task_manager.cpp
extern thread_barrier bar;

volatile extern int total_remain;
extern int futex_val;
void futex_wakeup();
void futex_sleep();

// This function allows a task to notify that it needs to initiate a system-wide
// re-schedule;
extern void initiate_reschedule();

// This function needs to be called at the start of #omp parallel in task.run, 
// in order to set up for mixed-criticality operation. This function needs the
// high criticality cpu mask.
void mode_change_setup();

// This function needs to be called at least once  at the end of #omp parallel 
// in task.run, in order to finish mixed-criticality operation. All work for 
// the current period must be finished before any thread calls this function. 
// This function may be called multiple times.
void mode_change_finish();

extern void allow_change();
extern bool modify_self(int new_mode);

//omp replacement thread pool
extern __uint128_t processor_A_mask;
extern __uint128_t processor_B_mask;
extern __uint128_t processor_C_mask;
extern __uint128_t processor_D_mask;

#endif /* RT_GOMP_TASK_H */
