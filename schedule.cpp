#include "schedule.h"
//const int INVALID_INDEX=-1;
#include <sys/types.h>
#include <unistd.h>
#include <cstddef>

Schedule::Schedule(std::string name_) : sharedMem(name_, READ_WRITE, MAXTASKS*sizeof(class TaskData))
{
	if(sharedMem::is_owner())
	{
		((struct overhead *) sharedMem::getOverhead())->utility.store((int)0);
	}
	else
	{
	}
}

Schedule::~Schedule() {}

TaskData * Schedule::add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_)
{
	int num = ((struct overhead *) sharedMem::getOverhead())->utility;
	((struct overhead *) sharedMem::getOverhead())->utility.store(num+1);
	//std::cout << getpid() << " making value "  <<  ((struct overhead *) sharedMem::getOverhead())->utility << std::endl;
	TaskData td(elasticity_, num_modes_, work_, span_, period_);
	((class TaskData *)sharedMem::getMapping())[num]=td;
	return &((class TaskData *)sharedMem::getMapping())[num];

}

class TaskData * Schedule::get_task(int n)
{
	if(!(n >= 0 && n <= ((struct overhead *) sharedMem::getOverhead())->utility))
	{
		return NULL;
		//fprintf(stderr, "Error. Invalid index. %d, %d\n",n,int(((struct overhead *) sharedMem::getOverhead())->utility));
	}

	return ((class TaskData *)sharedMem::getMapping())+n;
}

int Schedule::count(){
	return ((struct overhead *) sharedMem::getOverhead())->utility;
}

