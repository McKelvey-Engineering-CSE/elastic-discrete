#include "scheduler.h"
#include "taskData.h"
#include "print_module.h"

#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cerrno>
#include <float.h>
#include <map>
#include <tuple>

class Schedule * Scheduler::get_schedule(){
	return &schedule;
}

TaskData * Scheduler::add_task(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_){
	
	//add the task to the legacy schedule object, but also add to vector
	//to make the scheduler much easier to read and work with.
	auto taskData_object = schedule.add_task(elasticity_, num_modes_, work_, span_, period_, gpu_work_, gpu_span_, gpu_period_);

	task_table.push_back(std::vector<task_mode>());

	for (int j = 0; j < num_modes_; j++){

		task_mode item;
		item.cpuLoss = (1.0 / taskData_object->get_elasticity() * (std::pow(taskData_object->get_max_utilization() - (taskData_object->get_work(j) / taskData_object->get_period(j)), 2)));
		item.gpuLoss = 0;
		item.cores = taskData_object->get_CPUs(j);
		item.sms = taskData_object->get_GPUs(j);

		task_table.at(task_table.size() - 1).push_back(item);

	}

	//update the system TPC count
	maxSMS = taskData_object->get_total_TPC_count();

	return taskData_object;
}

//static vector size
std::vector<std::vector<Scheduler::task_mode>> Scheduler::task_table(100, std::vector<task_mode>(100));

//Implement scheduling algorithm
void Scheduler::do_schedule(size_t maxCPU){

	//dynamic programming table
	int N = task_table.size();
    std::vector<std::vector<std::vector<std::pair<double, double>>>> dp(N + 1, std::vector<std::vector<std::pair<double, double>>>(maxCPU + 1, std::vector<std::pair<double, double>>(maxSMS + 1, {100000, 100000})));
    std::map<std::tuple<int,int,int>, std::vector<int>> solutions;

	//First time through Make sure we have enough CPUs and GPUs
	//in the system and determine practical max for each task.	
	if (first_time) {

		int min_required_cpu = 0;
		int min_required_gpu = 0;

		//Determine minimum required processors
		for (int i = 0; i < schedule.count(); i++){
			
			//CPU first
			min_required_cpu += (schedule.get_task(i))->get_min_CPUs();
			(schedule.get_task(i))->set_CPUs_gained(0);

			//GPU next
			min_required_gpu += (schedule.get_task(i))->get_min_GPUs();
			(schedule.get_task(i))->set_GPUs_gained(0);

		}

		//Determine the practical maximum. This is how many are left after each task has been given its minimum.
		for (int i = 0; i < schedule.count(); i++){

			//CPU
			if ((NUMCPUS - min_required_cpu + (schedule.get_task(i))->get_min_CPUs()) < (schedule.get_task(i))->get_max_CPUs())
				(schedule.get_task(i))->set_practical_max_CPUs( NUMCPUS - min_required_cpu + (schedule.get_task(i))->get_min_CPUs());

			else
				(schedule.get_task(i))->set_practical_max_CPUs((schedule.get_task(i))->get_max_CPUs());

			//GPU
			if (((int)(maxSMS) - min_required_gpu + (schedule.get_task(i))->get_min_GPUs()) < (schedule.get_task(i))->get_max_GPUs())
				(schedule.get_task(i))->set_practical_max_GPUs( maxSMS - min_required_gpu + (schedule.get_task(i))->get_min_GPUs());

			else
				(schedule.get_task(i))->set_practical_max_GPUs((schedule.get_task(i))->get_max_GPUs());

		}
	}

	//Execute double knapsack algorithm
    for (size_t i = 1; i <= num_tasks; i++) {

        for (size_t w = 0; w <= maxCPU; w++) {

            for (size_t v = 0; v <= maxSMS; v++) {

                //invalid state
                dp[i][w][v] = {-1, -1};  
                
				//if the class we are considering is not allowed to switch modes
				//just treat it as though we did check it normally, and only allow
				//looking at the current mode.
				if (!(schedule.get_task(i - 1))->get_changeable()){

						auto item = task_table.at(i - 1).at((schedule.get_task(i - 1))->get_current_mode());

						//if item fits in both sacks
						if ((w >= item.cores) && (v >= item.sms) && (dp[i - 1][w - item.cores][v - item.sms].first != -1)) {

							int newCPULoss = dp[i - 1][w - item.cores][v - item.sms].first - item.cpuLoss;
							int newGPULoss = dp[i - 1][w - item.cores][v - item.sms].second - item.gpuLoss;
							
							//if found solution is better, update
							if ((newCPULoss + newGPULoss) > (dp[i][w][v].first + dp[i][w][v].second)) {

								dp[i][w][v] = {newCPULoss, newGPULoss};

								solutions[{i, w, v}] = solutions[{i - 1, w - item.cores, v - item.sms}];
								solutions[{i, w, v}].push_back((schedule.get_task(i - 1))->get_current_mode());

							}
						}
					}

				else {

					//for each item in class
					for (size_t j = 0; j < task_table.at(i - 1).size(); j++) {

						auto item = task_table.at(i - 1).at(j);

						//if item fits in both sacks
						if ((w >= item.cores) && (v >= item.sms) && (dp[i - 1][w - item.cores][v - item.sms].first != -1)) {

							int newCPULoss = dp[i - 1][w - item.cores][v - item.sms].first - item.cpuLoss;
							int newGPULoss = dp[i - 1][w - item.cores][v - item.sms].second - item.gpuLoss;
							
							//if found solution is better, update
							if ((newCPULoss + newGPULoss) > (dp[i][w][v].first + dp[i][w][v].second)) {

								dp[i][w][v] = {newCPULoss, newGPULoss};

								solutions[{i, w, v}] = solutions[{i - 1, w - item.cores, v - item.sms}];
								solutions[{i, w, v}].push_back(j);

							}
						}
					}
				}
     }
   }
 }

    //return optimal solution
	auto result = solutions[{N, maxCPU, maxSMS}];

	//update the tasks
	std::ostringstream mode_strings;
	print_module::buffered_print(mode_strings, "\n========================= \n", "New Schedule Layout:\n");
	for (size_t i = 0; i < result.size(); i++)
		print_module::buffered_print(mode_strings, "Task ", i, " is now in mode: ", result.at(i), "\n");
	print_module::buffered_print(mode_strings, "Total Loss from Mode Change: ", 100000 - dp[N][maxCPU][maxSMS].first, "\n=========================\n\n");
	print_module::flush(std::cerr, mode_strings);

	//this changes the number of CPUs each task needs for a given mode
	//(utilization)
	for (int i = 0; i < schedule.count(); i++)
		(schedule.get_task(i))->set_current_mode(result.at(i), false);

	//greedily give cpus on first run
	if (first_time) {

		int next_CPU = 1;

		//Actually assign CPUs to tasks. Start with 1.
		for (int i = 0; i < schedule.count(); i++){

			if ((schedule.get_task(i))->get_current_lowest_CPU() > 0){

				print_module::print(std::cerr, "Error in task ", i, ": all tasks should have had lowest CPU cleared. (this likely means memory was not cleaned up)\n");
				killpg(process_group, SIGKILL);
                return;

			}

			(schedule.get_task(i))->set_current_lowest_CPU(next_CPU);
			next_CPU += (schedule.get_task(i))->get_current_CPUs();

			if (next_CPU > num_CPUs + 1){

				print_module::print(std::cerr, "Error in task ", i, ": too many CPUs have been allocated.", next_CPU, " ", num_CPUs, " \n");
				killpg(process_group, SIGKILL);
				return;

			}		
		}

		//Now assign TPC units to tasks, same method as before
		int next_TPC = 1;

		for (int i = 0; i < schedule.count(); i++){

			if ((schedule.get_task(i))->get_current_lowest_GPU() > 0){

				print_module::print(std::cerr, "Error in task ", i, ": all tasks should have had lowest GPU cleared. (this likely means memory was not cleaned up)\n");
				killpg(process_group, SIGKILL);
				return;

			}

			(schedule.get_task(i))->set_current_lowest_GPU(next_TPC);
			next_TPC += (schedule.get_task(i))->get_current_GPUs();

			if (next_TPC > (int)(maxSMS) + 1){

				print_module::print(std::cerr, "Error in task ", i, ": too many GPUs have been allocated.", next_TPC, " ", maxSMS, " \n");
				killpg(process_group, SIGKILL);
				return;

			}		
		}
	}

	//Transfer as efficiently as possible.
	//This portion of code is supposed to allocate
	//CPUs from tasks that are not active to ones that
	//should be rescheduling right now... because of this,
	//it should also be a good candidate algorithm for
	//passing off the GPU SMs.
	else {

		//after the mode change, the previous holds what the tasks are actually using
		//the current holds what they should be using.
		for (int i = 0; i < schedule.count(); i++){
		
			int gained = (schedule.get_task(i))->get_current_CPUs() - (schedule.get_task(i))->get_previous_CPUs(); 
			(schedule.get_task(i))->set_CPUs_gained(gained);

		}

		//First determine how many CPUs each task gets from each other task.
		for (int i = 0; i < schedule.count(); i++){

			for (int j = i + 1; j < schedule.count(); j++){

				int task_gaining_cpus = ((schedule.get_task(i))->get_CPUs_gained() > 0 && (schedule.get_task(j))->get_CPUs_gained() < 0) ? i : j;
				int task_giving_cpus = ((schedule.get_task(j))->get_CPUs_gained() > 0 && (schedule.get_task(i))->get_CPUs_gained() < 0) ? i : j;

				//this condition can be false if both tasks are gaining CPUs or both are losing CPUs
				if (task_gaining_cpus != task_giving_cpus){

					int difference = abs((schedule.get_task(task_gaining_cpus))->get_CPUs_gained()) - abs((schedule.get_task(task_giving_cpus))->get_CPUs_gained());
					
					int amount = (difference >= 0) ? abs((schedule.get_task(task_giving_cpus))->get_CPUs_gained()) : abs((schedule.get_task(task_gaining_cpus))->get_CPUs_gained());

					(schedule.get_task(task_gaining_cpus))->set_CPUs_gained((schedule.get_task(task_gaining_cpus))->get_CPUs_gained() - amount);
					(schedule.get_task(task_giving_cpus))->set_CPUs_gained((schedule.get_task(task_giving_cpus))->get_CPUs_gained() + amount);
					(schedule.get_task(task_gaining_cpus))->update_give(task_giving_cpus, -1 * amount);
					(schedule.get_task(task_giving_cpus))->update_give(task_gaining_cpus, amount);
					
				}
			}
		}
		
		//Now determine which CPUs are transfered.
		for (int i = 0; i < schedule.count(); i++){

			for (int j = i + 1; j < schedule.count(); j++){

				cpu_set_t overlap;

				int task_giving_cpus = -1;
				int task_receiving_cpus = -1;

				//if task "i" is supposed to be giving CPUs
				if ((schedule.get_task(i))->gives(j) > 0){
					
					//comment with code
					task_giving_cpus = i;
					task_receiving_cpus = j;

				}

				//if task "j" is supposed to be giving CPUs
				else if ((schedule.get_task(i))->gives(j) < 0){

					//comment with code
					task_giving_cpus = j;
					task_receiving_cpus = i;

				}

				//if we have a task that is supposed to be giving CPUs
				if (task_giving_cpus != -1 && task_receiving_cpus != -1){

					//how many CPUs we are supposed to be giving
					int amount_gives = (schedule.get_task(task_giving_cpus))->gives(task_receiving_cpus);

					CPU_ZERO(&overlap);
				
					//determine which CPUs are active in i but passive in j
					for (int t = 1; t <= NUMCPUS; t++)
						if (schedule.get_task(task_giving_cpus)->get_active_cpu(t) && schedule.get_task(task_receiving_cpus)->get_passive_cpu(t))
							CPU_SET(t, &overlap);
	
					int amount_overlap = CPU_COUNT(&overlap);

					//if there are CPUs that are active in task giving but passive in task receiving
					if (amount_overlap > 0){

						for (int cpu_in_question = 1; cpu_in_question <= NUMCPUS; cpu_in_question++){
							
							//if cpu_in_question is in overlap set
							if (CPU_ISSET(cpu_in_question, &overlap)){

								bool used = false;

								//check if cpu_in_question is already being transferred
								for (int l = 0; l < schedule.count(); l++)
									if (schedule.get_task(task_giving_cpus)->transfers(l, cpu_in_question))
										used = true;
								
								//if not already transferred and not the permanent CPU, mark for transfer to task receiving CPUs
								if (!used && cpu_in_question != (schedule.get_task(task_giving_cpus))->get_permanent_CPU() && !(schedule.get_task(task_giving_cpus))->transfers(task_receiving_cpus, cpu_in_question)){
									
									(schedule.get_task(task_giving_cpus))->set_transfer(task_receiving_cpus, cpu_in_question, true);

									(schedule.get_task(task_receiving_cpus))->set_receive(task_giving_cpus, cpu_in_question, true);

									print_module::print(std::cerr, "Task ", task_giving_cpus, " should be sending CPU ", cpu_in_question, " to task ", task_receiving_cpus, ".\n");
									
									amount_gives--;

									if (amount_gives == 0)
										break;
								}
							}
						}
					}

					//if we still have cpus to give
					if (amount_gives > 0){

						for (int cpu_in_question = NUMCPUS; cpu_in_question >= 1; cpu_in_question--){

							//if cpu_in_question under consideration is active in giving task
							if (schedule.get_task(task_giving_cpus)->get_active_cpu(cpu_in_question)){

								bool used = false;
								
								//check if cpu_in_question is already being transferred
								for (int l = 0; l < schedule.count(); l++)
									if (schedule.get_task(task_giving_cpus)->transfers(l, cpu_in_question))
										used = true;
								
								//if not already transferred and not the permanent CPU, mark for transfer to receiving task
								if (!used && cpu_in_question != (schedule.get_task(task_giving_cpus))->get_permanent_CPU() && !(schedule.get_task(task_giving_cpus))->transfers(task_receiving_cpus, cpu_in_question)){

									print_module::print(std::cerr, "Task ", task_giving_cpus, " should be sending CPU ", cpu_in_question, " to task ", task_receiving_cpus, ".\n");

									(schedule.get_task(task_giving_cpus))->set_transfer(task_receiving_cpus, cpu_in_question, true);
									
									(schedule.get_task(task_receiving_cpus))->set_receive(task_giving_cpus, cpu_in_question, true);

									amount_gives--;

									if (amount_gives == 0)
										break;
                                }
							}
						}
					}
				}
			}			
		}
	}

	first_time = false;

}

void Scheduler::setTermination(){
	schedule.setTermination();
}