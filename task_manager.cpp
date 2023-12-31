// Each real time task should be compiled as a separate program and include task_manager.cpp and task.h
// in compilation. The task struct declared in task.h must be defined by the real time task.
//#define _GNU_SOURCE
#include <stdint.h> //For uint64_t
#include <stdlib.h> //For malloc
#include <sched.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <thread> //for std::thread::hardware_concurrency()
#include <sstream>
#include <vector>
#include <signal.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include "task.h"
#include "schedule.h"
#include "single_use_barrier.h"
#include "timespec_functions.h"
#include <signal.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <atomic> //For std::atomic_bool
#include <list>
//#include "futexold.h"
#include "bar.h" 
#include <sys/types.h>
#include <sys/syscall.h>
#include <limits.h> //for INT MAX
#include <map>
//#include "taskData.h"
//#include <map>
#include "include.h"

#define gettid() syscall(SYS_gettid)

extern const int NUMCPUS;
extern const int MAXTASKS;

//#define TRACING
//#define DAVID

extern "C" 
{
#include "dl_syscalls.h"
}

using namespace std;

#ifdef TRACING
FILE * fd;
#endif

//extern const unsigned int NUMCPUS;//There is always 1 CPU set aside for the scheduler object.

//mode_structure ms("apartec_ms");

//There are one trillion nanoseconds in a second, or one with nine zeroes
const unsigned nsec_in_sec = 1000000000; 

//The priority that we finalize the program with
const unsigned FINALIZE_PRIORITY = 1;

//The priority that we use when sleeping.
const unsigned SLEEP_PRIORITY = 97;


//These variables are declared extern in task.h, but need to be
//visible in both places
int futex_val;
mc_barrier bar;
bool missed_dl=false;
int num_threads;
volatile int total_remain __attribute__ (( aligned (64) ));
double percentile = 1.0;

int ret_val; //This value is used as a return value for system calls

//The process group ID is used to notify other tasks in this taskset that
//it is time to switch to high criticality mode
pid_t process_group;

pid_t mypid;

cpu_set_t global_cpuset;
struct sched_param global_param;

enum rt_gomp_task_manager_error_codes
{ 
	RT_GOMP_TASK_MANAGER_SUCCESS,
	RT_GOMP_TASK_MANAGER_CORE_BIND_ERROR,
	RT_GOMP_TASK_MANAGER_SET_PRIORITY_ERROR,
	RT_GOMP_TASK_MANAGER_INIT_TASK_ERROR,
	RT_GOMP_TASK_MANAGER_RUN_TASK_ERROR,
	RT_GOMP_TASK_MANAGER_BARRIER_ERROR,
	RT_GOMP_TASK_MANAGER_BAD_DEADLINE_ERROR,
	RT_GOMP_TASK_MANAGER_ARG_PARSE_ERROR,
	RT_GOMP_TASK_MANAGER_ARG_COUNT_ERROR
};

time_t start_sec=0, end_sec=0;//, min_work_sec=0, max_work_sec=0, span_sec=0, tmin_sec=0, tmax_sec=0, deadline_sec=0, relative_release_sec=0;
long start_nsec=0, end_nsec=0;//, min_work_nsec=0, max_work_nsec=0, span_nsec=0, tmin_nsec=0, tmax_nsec=0, deadline_nsec=0, relative_release_nsec=0;;
timespec start_time, end_time, current_period, current_work, current_span, deadline;//, min_work, max_work, span, tmin, tmax, deadline, relative_release;
//double elastic_coefficient=1.0;
//int rtprio=1;
int iindex=-1;
int current_mode = -1;

unsigned num_iters=0;

// Let scheduler know we need a rescheduling.
// Do this by sending signal SIGRTMIN+0.
void initiate_reschedule()
{
        killpg(process_group, SIGRTMIN+0);
}

cpu_set_t current_cpu_mask;
struct sched_param sp;
bool active[64];
bool needs_reschedule=false;

mutex con_mut;
Schedule schedule(std::string("EFSschedule"));

//has pthread_t at omp_thread index
vector<pthread_t> threads;
//tells which thread is on a CPU
map<int, pthread_t> thread_locations;

//tells OMP number of thread
map<pthread_t, int> omp_thread_index;

//unordered_map<pthread_t, bool> active_map;

void sigrt0_handler(int signum){
}

void reschedule();

void sigrt1_handler(int signum){
	needs_reschedule = true;
}

void modify_self(int new_mode){
	schedule.get_task(iindex)->set_current_mode(new_mode,true);
	killpg(process_group, SIGRTMIN+0);
}

void allow_change(){
	schedule.get_task(iindex)->reset_changeable();
	killpg(process_group, SIGRTMIN+0);
}

void reschedule()
{
	// Set up everything to begin as scheduled.
    current_period = schedule.get_task(iindex)->get_current_period();
    current_work = schedule.get_task(iindex)->get_current_work();
	current_mode = schedule.get_task(iindex)->get_current_mode();
	deadline = current_period;
	percentile = schedule.get_task(iindex)->get_percentage_workload();

	for(int i=0;i<schedule.count();i++)
	{
		for(int j=1; j<=NUMCPUS; j++)
		{
			if(schedule.get_task(iindex)->transfers(i,j))
			{
				schedule.get_task(iindex)->clr_active(j);
				//fprintf(stderr,"BLAHHH\n");
				schedule.get_task(iindex)->set_passive(j);
				active[omp_thread_index[thread_locations[j]]] = false;
	
				global_param.sched_priority=SLEEP_PRIORITY;
	                        ret_val = pthread_setschedparam(thread_locations[j], SCHED_RR, &global_param);
				if(ret_val < 0)
				{fprintf(stderr,"ERROR: could not set priority.\n");}
				//fprintf(stderr,"MADE IT OUT!!");
			}
			else if (schedule.get_task(iindex)->receives(i,j))
			{
				//fprintf(stderr,"BLAHHH2\n");	
				if(schedule.get_task(iindex)->get_passive(j))
				{
					//fprintf(stderr,"BLAHHH3\n");
					//CPU_CLR(j,schedule.get_task(iindex)->get_passive());
                                	//CPU_SET(j,schedule.get_task(iindex)->get_active());
					schedule.get_task(iindex)->clr_passive(j);
					schedule.get_task(iindex)->set_active(j);

					global_param.sched_priority=7;
					ret_val = pthread_setschedparam(thread_locations[j], SCHED_RR, &global_param);
					
					active[omp_thread_index[thread_locations[j]]] = true;
				}
				else
				{	
					//fprintf(stderr,"%dBLAHHH4\n",iindex);
					for(int k = NUMCPUS; k>=1 ; k--)
					{
						if(schedule.get_task(iindex)->get_passive(k))
						{
							for(int i=1; i<=NUMCPUS; i++)
							{
								std::cout << active[i] << std::endl;
							}	
							std::cout << omp_thread_index[thread_locations[k]] << " " << thread_locations[k]  << " "  <<  k << std::endl;							

							active[omp_thread_index[thread_locations[k]]] = true;
							schedule.get_task(iindex)->clr_passive(k);
							//schedule.get_task(iindex)->set_active(k);	
	
							thread_locations[j]=thread_locations[k];
							thread_locations.erase(k);

							active[omp_thread_index[thread_locations[j]]] = true;
							schedule.get_task(iindex)->set_active(j); 
							global_param.sched_priority=7;
							ret_val = pthread_setschedparam(thread_locations[j], SCHED_RR, &global_param);	
							CPU_ZERO(&global_cpuset);
							CPU_SET(j,&global_cpuset);
							pthread_setaffinity_np(thread_locations[j],sizeof(cpu_set_t),&global_cpuset);

							break;
						}
					}
				}
			}	
		}
	}		

	mc_bar_reinit(&bar,schedule.get_task(iindex)->get_current_CPUs());		
}

//typedef struct sched_attr sched_attr;

int main(int argc, char *argv[])
{
	fflush(stdout);
	fflush(stderr);

	#ifdef TRACING
	fd = fopen( "/sys/kernel/debug/tracing/trace_marker", "a" );
	if( fd == NULL ){
		perror("Error: TRACING is defined and you are not using trace-cmd.");
		return -1;
	}
	#endif


	std::string command_string;	
	for( int i = 0; i < argc; i++ ){
		command_string += " ";
		command_string += argv[i];
	}

	fprintf(stderr, "Task %d started with command string:\n>%s\n", getpid(), command_string.c_str());

	//Get our own PID and store it
	mypid = getpid();

	//Determine what our current process group is, so we can notify the
	//system if we miss our virtual deadline
	process_group = getpgrp();

	//Set up a signal handler for SIGRT0, this manages the notification of
	//the system high crit transition
	void (*ret_handler)(int);
	ret_handler = signal(SIGRTMIN+0, sigrt0_handler);
	if( ret_handler == SIG_ERR ){
		fprintf(stderr,"ERROR: Call to Signal failed, reason: %s\n", strerror(errno));
		exit(-1);
	}

	ret_handler = signal(SIGRTMIN+1, sigrt1_handler);
	if( ret_handler == SIG_ERR ){
		fprintf(stderr,"ERROR: Call to Signal failed, reason: %s\n", strerror(errno));
		exit(-1);
	}

	// Process command line arguments	
	const char *task_name = argv[0];

	if(!((std::istringstream(argv[1]) >> start_sec) &&
		(std::istringstream(argv[2]) >> start_nsec) &&
		(std::istringstream(argv[3]) >> end_sec) &&
        (std::istringstream(argv[4]) >> end_nsec) &&
		(std::istringstream(argv[5]) >> iindex)))
		
	{
		fprintf(stderr, "ERROR: Cannot parse input argument for task %s", task_name);
                kill(0, SIGTERM);
                return RT_GOMP_TASK_MANAGER_ARG_PARSE_ERROR;
	}

	start_time = {start_sec, start_nsec};
	end_time = {end_sec, end_nsec};
	
	char *barrier_name = argv[6];
	int task_argc = argc - 7;                                             
	char **task_argv = &argv[7];

	//fprintf(stderr, "%s : %d : %s\n",barrier_name, task_argc, task_argv[0]); 
	//fprintf(stderr, "Task %s reached scheduling barrier\n", task_name);
	//Wait at barrier for the other tasks but mainly to make sure scheduler has finished
	ret_val = await_single_use_barrier("RT_GOMP_CLUSTERING_BARRIER2");
	if (ret_val != 0)
	{
		fprintf(stderr, "ERROR: Barrier error for task %s", task_name);
		kill(0, SIGTERM);
		return RT_GOMP_TASK_MANAGER_BARRIER_ERROR;
	}

	// Set up everything to begin as scheduled.
	current_work = schedule.get_task(iindex)->get_current_work();
	current_period = schedule.get_task(iindex)->get_current_period();
	deadline = current_period;
	percentile = schedule.get_task(iindex)->get_percentage_workload();

	CPU_ZERO(&current_cpu_mask);
	for(int i = 0; i < schedule.get_task(iindex)->get_current_CPUs(); i++ )
	{
		CPU_SET(schedule.get_task(iindex)->get_current_lowest_CPU() + i,&current_cpu_mask);
	}
	
	{
	lock_guard<mutex> lk(con_mut);
	fprintf(stderr, "Task %d lowest CPU: %d, currentCPUS, %d minCPUS: %d, maxCPUS: %d, practical max: %d, %ldns\n",getpid(), schedule.get_task(iindex)->get_current_lowest_CPU(), schedule.get_task(iindex)->get_current_CPUs(), schedule.get_task(iindex)->get_min_CPUs(), schedule.get_task(iindex)->get_max_CPUs(), schedule.get_task(iindex)->get_practical_max_CPUs(), schedule.get_task(iindex)->get_current_period().tv_nsec);
	}

	struct sched_param param;
	param.sched_priority = 7;//rtprio;
	ret_val = sched_setscheduler(getpid(), SCHED_RR, &param);
	if (ret_val != 0)
	{
 		fprintf(stderr,"WARNING: %d  Could not set priority. Returned %d (%s)\n", getpid(), errno, strerror(errno));
	}

	// OMP settings
	omp_set_nested(0);
	omp_set_dynamic(0);
	omp_set_schedule(omp_sched_dynamic, 1);	
	//num_threads = schedule.get_task(iindex)->get_max_CPUs();
	num_threads = schedule.get_task(iindex)->get_practical_max_CPUs();
	omp_set_num_threads(num_threads);

	for(int t=1; t<=NUMCPUS; t++){
		schedule.get_task(iindex)->clr_active(t);
		schedule.get_task(iindex)->clr_passive(t);
	}

	for(unsigned int j=0;j<std::thread::hardware_concurrency();j++)
	{
		active[j] = false;
	}
 
	threads.reserve(NUMCPUS);
	#pragma omp parallel
	{
		//lock_guard<mutex> lk(con_mut);
		threads[omp_get_thread_num()]=pthread_self();
		omp_thread_index[pthread_self()]=omp_get_thread_num();	
	}


	//Determine which threads are active and passive. Pin, etc.
	for(int j=0;j<num_threads;j++)
  	{
		//The first threads are active
		if(j < schedule.get_task(iindex)->get_current_CPUs())
		{
			active[j]=true;
			//What processor does it go on?
			int p = (schedule.get_task(iindex)->get_current_lowest_CPU() + j -1) % (NUMCPUS) + 1;

			//First one doesn't move.  
			if(j==0)
			{
				schedule.get_task(iindex)->set_permanent_CPU(p);
			}
				
			global_param.sched_priority=7;
            ret_val = pthread_setschedparam(threads[j], SCHED_RR, &global_param);
			thread_locations[p]=threads[j];
			fprintf(stderr,"%d setting %d to active.\n",iindex,p);
			schedule.get_task(iindex)->set_active(p);
			CPU_ZERO(&global_cpuset);
			CPU_SET(p,&global_cpuset);
			pthread_setaffinity_np(threads[j],sizeof(cpu_set_t),&global_cpuset);
 
		}
		//We're done with the Active threads. Rest are passive.
		else
		{
			active[j] = false;
			//What processor does it go on?
            int p = (schedule.get_task(iindex)->get_current_lowest_CPU() + j -1) % (NUMCPUS) + 1;
			global_param.sched_priority=SLEEP_PRIORITY;
                        ret_val = pthread_setschedparam(threads[j], SCHED_RR, &global_param);	
			thread_locations[p]=threads[j];
			schedule.get_task(iindex)->set_passive(p);
			fprintf(stderr,"%d setting %d to passive.\n",iindex,p);
			CPU_ZERO(&global_cpuset);
			CPU_SET(p,&global_cpuset);
			pthread_setaffinity_np(threads[j],sizeof(cpu_set_t),&global_cpuset);

		}
  	}
		
	//Initialize the program barrier
	mc_bar_init(&bar,schedule.get_task(iindex)->get_current_CPUs());

	#ifdef PER_PERIOD_VERBOSE
	//Create storage for per-job timings --THIS NO LONGER WORKS!! num_iters is no longer static --James 9/8/18
	uint64_t *period_timings = (uint64_t*) malloc(num_iters * sizeof(uint64_t));
	#endif
	
	// Initialize the task
	if(schedule.get_task(iindex) && task.init != NULL)
	{
		ret_val = task.init(task_argc, task_argv);
		if (ret_val != 0)
		{
			fprintf(stderr, "ERROR: Task initialization failed for task %s", task_name);
			kill(0, SIGTERM);
			return RT_GOMP_TASK_MANAGER_INIT_TASK_ERROR;
		}
	}
	
	fprintf(stderr, "Task %s reached barrier\n", task_name);
	// Wait at barrier for the other tasks
	ret_val = await_single_use_barrier(barrier_name);
	if (ret_val != 0)
	{
		fprintf(stderr, "ERROR: Barrier error for task %s", task_name);
		kill(0, SIGTERM);
		return RT_GOMP_TASK_MANAGER_BARRIER_ERROR;
	}
	
	//Don't go until it's time.
	timespec current_time;
	do {
		 get_time(&current_time);
	}while(current_time < start_time);
	//Grab a timestamp at the start of real-time operation

	// Initialize timing controls
	unsigned deadlines_missed = 0;
	timespec correct_period_start, actual_period_start, period_finish, period_runtime;
	get_time(&correct_period_start);
	timespec max_period_runtime = { 0, 0 };
	uint64_t total_nsec = 0;
	//timespec tmp1, tmp2, tmp3;

	while(current_time < end_time)	
	{
		//fprintf(stderr, "current time {%lds,%ldns}. end time {%lds,%ldns}",current_time.tv_sec,current_time.tv_nsec,end_time.tv_sec,end_time.tv_nsec);
		if(schedule.get_task(iindex))
		{
			num_iters++;
		}     	

		// Sleep until the start of the period
		sleep_until_ts(correct_period_start);

        #ifdef TRACING
		fprintf( fd, "thread %ld: starting iteration %d\n", gettid() ,num_iters);
		fflush( fd );
        #endif

		//when we start doing the work
   		get_time(&actual_period_start);

		// Reset the awaited count of threads before every period
		__atomic_store_n( &total_remain, num_threads, __ATOMIC_RELEASE ); //JAMES ORIGINAL 10/3/17

		if(schedule.get_task(iindex))
		{
			ret_val = task.run(task_argc, task_argv);
		}

		//Get the finishing time of the current period
		get_time(&period_finish);
		if (ret_val != 0)
		{
			fprintf(stderr, "ERROR: Task run failed for task %s", task_name);
			return RT_GOMP_TASK_MANAGER_RUN_TASK_ERROR;
		}
		
		// Check if the task finished before its deadline and record the maximum running time
		ts_diff(correct_period_start, period_finish, period_runtime);
		if (period_runtime > deadline) 
		{
			deadlines_missed += 1;
			missed_dl=true;

			#ifdef TRACING
			fprintf( fd, "thread %d: missed deadline iteration %d\n", getpid() ,num_iters);
			fflush( fd );
            #endif
		}
		else
		{
			missed_dl=false;
		}
		if (period_runtime > max_period_runtime) max_period_runtime = period_runtime;
		total_nsec += period_runtime.tv_nsec + nsec_in_sec * period_runtime.tv_sec;

		// Update the period_start time
		correct_period_start = correct_period_start + current_period;

		
		if(needs_reschedule)
		{       
			bool ready = true;
			//Check other tasks to see if this task can transition yet. It can iff it is giving up a CPU, or is gaining a CPU that has been given up.
			for(int i=0; i<schedule.count(); i++)
			{
				for(int j=1; j<=NUMCPUS; j++)
				{
					if ((schedule.get_task(i)->transfers(iindex,j)))
					{
						//fprintf(stderr,"I'm task %d and I'm supposed to be getting CPU %d from task %d\n",iindex,j,i);
						if((schedule.get_task(i))->get_num_adaptations() <=  (schedule.get_task(iindex))->get_num_adaptations())
						{
							ready = false;
						}
					}
     			}
			}

			if(ready)
			{
				#ifdef TRACING
				fprintf( fd, "thread %d: starting reschedule\n", getpid());
				fflush( fd );
				#endif

				reschedule();
				
				fprintf( stderr, "thread %d: finished reschedule\n", iindex);	
				#ifdef TRACING
				fprintf( fd, "thread %d: finished reschedule\n", getpid());
				fflush( fd );
				#endif
				schedule.get_task(iindex)->set_num_adaptations(schedule.get_task(iindex)->get_num_adaptations()+1);
				needs_reschedule = false;
			}
			else
			{
				//Gaining a processor that wasn't ready yet.
				fprintf(stderr,"task %d can't reschedule!\n",getpid());
			}
		}

		get_time(&current_time);
		
	}

	// Lower priority as soon as we're done running in real-time
	sp.sched_priority = FINALIZE_PRIORITY;
	ret_val = sched_setscheduler(getpid(), SCHED_RR, &sp);
	
	if (ret_val != 0)
	{
		perror("WARNING: Could not set FINALIZE_PRIORITY");
	}
	
	// Finalize the task
	#ifdef TRACING
	fclose(fd);
	#endif
	if (schedule.get_task(iindex) && task.finalize != NULL) 
	{
		ret_val = task.finalize(task_argc, task_argv);
		if (ret_val != 0)
		{
			fprintf(stderr, "WARNING: Task finalization failed for task %s\n", task_name);
		}
	}

	//Print out useful information.
	fprintf(stderr,"(%d) Deadlines missed for task %s: %d/%d\n", mypid, task_name, deadlines_missed, num_iters);
	fprintf(stderr,"(%d) Max running time for task %s: %i sec  %lu nsec\n", mypid, task_name, (int)max_period_runtime.tv_sec, max_period_runtime.tv_nsec);
	fprintf(stderr,"(%d) Avg running time for task %s: %0.6f  msec\n", mypid, task_name, (total_nsec/(num_iters))/1000000.0);

	
	#ifdef PER_PERIOD_VERBOSE
	std::ofstream outfile;
	outfile.open("time_results.txt", std::ios::out | std::ios::app);
	outfile << omp_get_num_procs();
	for(unsigned i = 0; i < num_iters; i++){
		outfile << " " << period_timings[i];
	}
	outfile << std::endl;
	outfile.close();
	#endif
	
	std::cout << deadlines_missed << " " << num_iters << " " << omp_get_num_threads() << " " << max_period_runtime << std::endl;

	fflush(stdout);
	fflush(stderr);

return 0;
}	
