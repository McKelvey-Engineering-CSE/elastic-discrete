#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <sys/types.h>
#include <unistd.h>

#include <sched.h>


#include "dl_syscalls.h"

typedef struct sched_attr sched_attr;

int main(int argc, char* argv[]){

	int ret; //return value

	if(argc != 2){
		printf("This program takes one argument that specifies whether it should try to set CPU affinity. If the argument is 0 it does not set affinity, otherwise it will.\n");
		exit(-1);
	}
	int set_affinity = atoi(argv[1]);


	if( set_affinity != 0 ){
	/*First we restrict our eligible range of processors*/

	printf("Attempting to set CPU affinity...\n");

	//The sched_setaffinity method does not work with the SCHED_DEADLINE scheduler
	/*	
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(3,&mask);
	
	ret=0;
	ret = sched_setaffinity( getpid(), sizeof(mask), &mask);
	if( ret != 0 ){
		printf(" ERROR: Could not set cpu affinity. Reason: %s\n", strerror(errno));
		exit(-1);
	}
	*/

	//The cpuset method is supposed to work, however. This has been encapsulated in a script
	std::stringstream command;	
	pid_t myPID = getpid();
	command << "./check_cpuset 3 " << myPID;
	printf("Calling command: %s\n", command.str().c_str());
	ret = system(command.str().c_str());
	if( ret != 0 ){
		printf(" ERROR: Call to script check_cpuset failed!\n");
		exit(-1);
	}

	printf("Affinity set successfully!\n");
	}
	

	printf("Attempting to set SCHED_DEADLINE..."); 

	sched_attr attr;

	attr.size = sizeof(sched_attr);
	attr.sched_policy = SCHED_DEADLINE;
	attr.sched_flags = 0;
	attr.sched_nice = 0;
	attr.sched_priority = 0;
	attr.sched_runtime = 1000000;
	attr.sched_deadline = 1000000000;
	attr.sched_period = attr.sched_deadline;

	ret = 0;
	ret = sched_setattr(0, &attr, 0);
	if( ret != 0 ){
		printf(" ERROR: %s\n", strerror(errno));
		exit(-1);
	} else {
		printf(" Success!\n");
	}
	
	return 0;
}
