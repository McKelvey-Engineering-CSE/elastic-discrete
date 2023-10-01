#include <omp.h>
#include <sstream>
#include <stdio.h>
#include "task.h"
#include "timespec_functions.h"
#include <unistd.h>
#include <math.h>
#include <sched.h>
#include <sys/types.h>
#include <string.h>
#include <signal.h>

#include <stdint.h> //For uint64_t                
#include <stdlib.h> //For malloc
struct itimerspec disarming_its, virtual_dl_timer_its;
static timer_t vd_timer;

timespec vdeadline={0,0};
timespec zero={0,0};
//timespec new_value;
int new_mode=-1;

int ret=0;
pid_t mapid;
struct sigevent sev;

//This boolean value is true IFF you modify a task during execution.
bool adapt = false;


void arm_virtual_dl_timer(){
    	ret = timer_settime( vd_timer, 0, &virtual_dl_timer_its, NULL);
    	if ( ret == -1 )
    	{ fprintf(stderr,"WARNING: Could not arm virtual deadline timer. Reason: %s\n", strerror(errno)); }
}

void disarm_virtual_dl_timer(){
    ret = timer_settime( vd_timer, 0, &disarming_its, NULL);
    if ( ret == -1 )
    { fprintf(stderr,"WARNING: Could not disarm virtual deadline timer. Reason: %s\n", strerror(errno)); }
}

//Timer went off. Signal mode change                                                                                                   
void sigrt2_handler(int signum){
    //fprintf(stderr,"(%d) SIGRT2 caught, WCET timer has fired\n", mapid); 
	disarm_virtual_dl_timer();
	modify_self(new_mode);
}


int init (int argc, char* argv[])//JO, cpu_set_t *all_cpus_mask )
{
   	if (!(
          std::istringstream(argv[argc-1]) >> new_mode &&
          std::istringstream(argv[argc-2]) >> adapt
        ))
    	{   
        	fprintf(stderr, "ERROR: Cannot parse vdl input argument for task %s", argv[0]);
        	kill(0, SIGTERM);
        	return -1;
    	}

	if(adapt)
	{

    	vdeadline = {10, 0};
    	virtual_dl_timer_its.it_interval = zero;
    	virtual_dl_timer_its.it_value = vdeadline;
    	disarming_its.it_interval = zero;
    	disarming_its.it_value = zero;
    
    	sev.sigev_value.sival_ptr = &vd_timer;
    	sev.sigev_signo = SIGRTMIN+2;
    	sev.sigev_notify = SIGEV_SIGNAL;
    	sev.sigev_notify_function = NULL;
    	sev.sigev_notify_attributes = NULL;
    
    	ret = timer_create(CLOCK_MONOTONIC, &sev, &vd_timer);
    	if ( ret == -1 )
    	{   
        	fprintf(stderr,"ERROR: Could not create virtual deadline timer: %s\n", strerror(errno));
     		exit(-1);
    	}
    
    	void (*ret_handler)(int);
    	ret_handler = signal(SIGRTMIN+2, sigrt2_handler);
    	if( ret_handler == SIG_ERR ){
        	fprintf(stderr,"ERROR: Call to Signal failed, reason: %s\n", strerror(errno));
      		exit(-1);
    	}
	
		arm_virtual_dl_timer();	
	}
	
	return 0;
}


int run (int argc, char* argv[])
{

    if (argc < 1)
    {
        fprintf(stderr, "ERROR: Two few arguments\n");
	return -1;
    }
	    
    int taskargc = 1;
    int segarc = 3;

    int num_segments;
    if (!(std::istringstream(argv[1]) >> num_segments))
    {
        fprintf(stderr, "ERROR: Cannot parse input argument\n");
        return -1;
    }


//TODO:Find a way for this to change with the mode. 

#pragma omp parallel 
{

	mode_change_setup();


	// For each segment
	int i;
	for (i = 0; i < num_segments; ++i)
	{
		if (argc < 5  + segarc*i)
	    	{
	        	fprintf(stderr, "ERROR: Two few arguments\n");
			//return -1;
	    	}
	    
	    	int num_strands;
	    	long len_sec, len_ns;
	    	if (!(
	        	std::istringstream(argv[taskargc+1 + segarc*i]) >> num_strands &&
	        	std::istringstream(argv[taskargc+2 + segarc*i]) >> len_sec &&
	        	std::istringstream(argv[taskargc+3 + segarc*i]) >> len_ns ))
	    	{
	        	fprintf(stderr, "ERROR: Cannot parse input argument\n");
		    	//return -1;
	    	}
	    
		timespec segment_length = { len_sec, len_ns };
		
		// For each strand in parallel
		int j;
		#pragma omp for schedule(dynamic) nowait 
		for (j = 0; j < num_strands; ++j)
		{
			//std::cout << getpid() << " " <<  percentile << std::endl;
			busy_work(segment_length);
			//busy_work( percentile * segment_length);
		}
		mc_bar_wait(&bar);
		
		//fprintf(stderr,"(%d) CPU %d is done with first loop, time: %0.3f\n", mapid, thread_id, end-start);
	}

	mode_change_finish();
}

	return 0;
}

int finalize (int argc, char* argv[]){

	return 0;
}

task_t task = {init, run, finalize};

