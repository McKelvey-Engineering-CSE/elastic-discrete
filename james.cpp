#include "task.h"

int init(int argc, char *argv[])
{
	return 0;       
}

int run(int argc, char *argv[]){
	return 0;
}

int finalize(int argc, char *argv[])
{
    return 0;
}

task_t task = { init, run, finalize };
