#include "schedule.h"
#include <sys/types.h>
#include <unistd.h>
#include <cstddef>

Schedule::Schedule(std::string name_) : shared_mem(name_, READ_WRITE, MAXTASKS*sizeof(class TaskData))
{
	if(shared_mem::is_owner())
	{
		((struct overhead *) shared_mem::getOverhead())->utility.store((int)0);
	}
}

Schedule::~Schedule() {}

TaskData * Schedule::add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_)
{
	int num = ((struct overhead *) shared_mem::getOverhead())->utility;
	((struct overhead *) shared_mem::getOverhead())->utility.store(num+1);
	TaskData td(elasticity_, num_modes_, work_, span_, period_);
	((class TaskData *)shared_mem::getMapping())[num]=td;
	return &((class TaskData *)shared_mem::getMapping())[num];

}

class TaskData * Schedule::get_task(int n)
{
	if(!(n >= 0 && n <= ((struct overhead *) shared_mem::getOverhead())->utility))
	{
		return NULL;
	}

	return ((class TaskData *)shared_mem::getMapping())+n;
}

int Schedule::count(){
	return ((struct overhead *) shared_mem::getOverhead())->utility;
}

