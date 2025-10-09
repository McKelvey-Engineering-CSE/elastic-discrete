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

TaskData * Schedule::add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_, timespec * cpu_C_work_, timespec * cpu_C_span_, timespec * cpu_C_period_, timespec * gpu_D_work_, timespec * gpu_D_span_, timespec * gpu_D_period_, bool safe)
{

	//build the quivalency vector from the equivalent vector we have stored
	std::tuple<int, float> equivalent_vector_partial[4];
	for (int i = 0; i < 4; i++)
		equivalent_vector_partial[i] = equivalent_vector[i];

	//now remove the first 4 elements from the equivalent vector
	equivalent_vector.erase(equivalent_vector.begin(), equivalent_vector.begin() + 4);

	underlying_object->task[underlying_object->next_task++] = TaskData(elasticity_, num_modes_, work_, span_, period_, gpu_work_, gpu_span_, gpu_period_, cpu_C_work_, cpu_C_span_, cpu_C_period_, gpu_D_work_, gpu_D_span_, gpu_D_period_, safe, equivalent_vector_partial, print);

	print = false;

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

void Schedule::set_equivalency_vector(std::vector<std::tuple<int, float>> _equivalent_vector){
	
	for (int i = 0; i < _equivalent_vector.size(); i++)
		equivalent_vector.push_back(_equivalent_vector[i]);

}