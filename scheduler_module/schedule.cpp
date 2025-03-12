#include "schedule.h"
#include <sys/types.h>
#include <unistd.h>
#include <cstddef>

Schedule::Schedule(std::string name_, bool create)
{

	schedule_object *obj;

	if (create){

		obj = shared_memory_module::allocate<schedule_object>(name_);
		owner = true;

	}

	else
		obj = shared_memory_module::fetch<schedule_object>(name_);

	if (obj == nullptr){
		exit(-1);
	}
	
	underlying_object = obj;

	name = name_;

}

Schedule::~Schedule() {
	
	if (owner)
		smm::delete_memory<schedule_object>(name);
}

TaskData * Schedule::add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_)
{

	underlying_object->task[underlying_object->next_task++] = TaskData(elasticity_, num_modes_, work_, span_, period_, gpu_work_, gpu_span_, gpu_period_);

	return &underlying_object->task[underlying_object->next_task - 1];

}

class TaskData * Schedule::get_task(int n)
{

	return &underlying_object->task[n];

}

int Schedule::count(){
	return underlying_object->next_task - 1;
}

void Schedule::setTermination(){
	smm::delete_memory<schedule_object>(name);
}
