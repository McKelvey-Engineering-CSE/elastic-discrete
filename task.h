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
#include <atomic> //for atomic_bool
#include "bar.h" //for mc_barrier
#include "bar.h"
#include <sched.h> //for cpu_set_t
#include "include.h"

//#define TRACING

#ifdef TRACING
extern FILE * fd;
#endif

// Task struct type used by task_manager.cpp to control a task.
typedef struct
{
	int (*init)(int argc, char *argv[]);
	int (*run)(int argc, char *argv[]);
	int (*finalize)(int argc, char *argv[]);
}
task_t;

extern bool active[64];
extern int current_mode;
extern double percentile;

extern void modify_self(timespec new_value);

// Used to determine current task and its features.
extern bool missed_dl;

// Task struct that should be defined by the real time task.
extern task_t task;

extern const int NUMCPUS;
extern const int MAXTASKS;

extern timespec current_period;
extern timespec current_work;

// We need a special barrier that understands the mixed-criticality mode
// transition and the fact that different numbers of threads are expected at
// the barrier at different times. This provides that, and should be used
// instead of any other barrier. Initialized in task_manager.cpp
extern cppBar bar;

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
extern void modify_self(int new_mode);
#endif /* RT_GOMP_TASK_H */
