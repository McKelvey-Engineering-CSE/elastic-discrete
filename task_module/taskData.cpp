#include "taskData.h"	

int TaskData::counter=0;	        

int TaskData::get_index(){
	return index;
}

int TaskData::get_CPUs_gained(){
	return CPUs_gained;
}

void TaskData::set_CPUs_gained(int new_CPUs_gained){
	CPUs_gained = new_CPUs_gained;
}

int TaskData::get_previous_CPUs(){
	return previous_CPUs;
}

void TaskData::set_previous_CPUs(int new_prev){
	previous_CPUs = new_prev;
}

void TaskData::update_give(int index, int value){
	give[index]=value;
}

int TaskData::gives(int index){
	return give[index];
}

timespec TaskData::get_max_work(){
	return max_work;
}

double TaskData::get_elasticity(){
	return elasticity;
}

double TaskData::get_max_utilization(){
	return max_utilization;
}

double TaskData::get_min_utilization(){
	return min_utilization;
}

int TaskData::get_max_CPUs(){
	return max_CPUs;
}

int TaskData::get_min_CPUs(){
	return min_CPUs;
}

timespec TaskData::get_current_work(){
	return current_work;
}	

timespec TaskData::get_current_period(){
	return current_period;
}

double TaskData::get_current_utilization(){
	return current_utilization;
}

int TaskData::get_current_CPUs(){
	return current_CPUs;
}

double TaskData::get_percentage_workload(){
	return percentage_workload;
}

bool TaskData::get_changeable(){
	return changeable;
}

int TaskData::get_current_lowest_CPU(){
	return current_lowest_CPU;
}

double TaskData::get_practical_max_utilization(){
	return practical_max_utilization;
}

void TaskData::set_practical_max_CPUs(int new_value){
	practical_max_CPUs = new_value;
}

	int TaskData::get_practical_max_CPUs(){
	return practical_max_CPUs;
}

void TaskData::set_current_lowest_CPU(int _lowest){
	current_lowest_CPU = _lowest;
}

void TaskData::set_active(int i){
	active[i]=true;
}

void TaskData::clr_active(int i){
			active[i]=false;
	}

void TaskData::set_passive(int i){
	passive[i]=true;
}

void TaskData::clr_passive(int i){
	passive[i]=false;
}

bool TaskData::get_active(int i){
	return active[i];
}

bool TaskData::get_passive(int i){
	return passive[i];
}

void TaskData::set_current_mode(int new_mode, bool disable)
{
	if(new_mode>=0 && new_mode<num_modes)
	{			
		current_mode = new_mode;
		current_work = work[current_mode];
		current_span = span[current_mode];
		current_period = period[current_mode];
		current_utilization = current_work/current_period;
		percentage_workload = current_work/max_work;
		previous_CPUs = current_CPUs;
		current_CPUs = CPUs[current_mode];
		if(disable)
		{
			changeable = false;
		}
	}
	else
	{
		print(std::cerr, "Error: Task ", get_index(), " was told to go to invalid mode ", new_mode, ". Ignoring.\n");
	}
}

int TaskData::get_current_mode()
{
	return current_mode;
}

void TaskData::reset_changeable(){
	changeable = true;
}

bool TaskData::transfers(int task, int CPU){
	return transfer[task][CPU];
}
	void TaskData::set_transfer(int task, int CPU, bool value){
	transfer[task][CPU]=value;
}

bool TaskData::receives(int task, int CPU){
	return receive[task][CPU];
}
	void TaskData::set_receive(int task, int CPU, bool value){
	receive[task][CPU]=value;
}

	int TaskData::get_permanent_CPU(){
	return permanent_CPU;
}
	void TaskData::set_permanent_CPU(int perm){
	permanent_CPU = perm;
}

	int TaskData::get_num_adaptations(){
	return num_adaptations;
}
	void TaskData::set_num_adaptations(int new_num){
	num_adaptations = new_num;
}

int TaskData::get_num_modes(){
	return num_modes;
}

timespec TaskData::get_work(int index){
	return work[index];
}
timespec TaskData::get_span(int index){
	return span[index];
}
timespec TaskData::get_period(int index){
	return period[index];
}
int TaskData::get_CPUs(int index){
	return CPUs[index];
}


