#include "task.h"
#include "ported_standard_library.hpp"

int init(int argc, char *argv[])
{
	return 0;       
}

int run(int argc, char *argv[]){
	//*(int * ) 0 = 0;
	return 0;
}

int finalize(int argc, char *argv[])
{
    return 0;
}

task_t task = { init, run, finalize };
