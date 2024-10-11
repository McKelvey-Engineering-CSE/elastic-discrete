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
#include <queue>

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

//static vector sizes
std::vector<std::vector<Scheduler::task_mode>> Scheduler::task_table(100, std::vector<task_mode>(100));
std::vector<Scheduler::task_mode> Scheduler::previous_modes(100, task_mode());

//this function compares the previous state of the system (current running modes)
//and checks each mode supplied by the scheduler to see if it is vulnerable.
//If it isn't, then it is placed at the front of the list. If it is, then it is
//placed at the back of the list. All items that only take resources are placed
//in at the very end of the list at the end of the sort
std::vector<int> Scheduler::sort_classes(std::vector<int> items_in_candidate) {
    
    std::vector<int> class_index_mappings = {-1};
	std::vector<int> back_inserted = {-1};
    
    //for all items in the class
    for (size_t i = 0; i < task_table.size(); i++){
		
		//if vulnerable
		if (((task_table.at(i).at(items_in_candidate.at(i)).cores - previous_modes.at(i).cores) > 0) && ((task_table.at(i).at(items_in_candidate.at(i)).sms - previous_modes.at(i).sms) < 0))
			class_index_mappings.push_back(i);
		
		//if vulnerable
		else if (((task_table.at(i).at(items_in_candidate.at(i)).cores - previous_modes.at(i).cores) < 0) && ((task_table.at(i).at(items_in_candidate.at(i)).sms - previous_modes.at(i).sms) > 0))
			class_index_mappings.push_back(i);

		//if state only takes
		else if (((task_table.at(i).at(items_in_candidate.at(i)).cores - previous_modes.at(i).cores) <= 0) && ((task_table.at(i).at(items_in_candidate.at(i)).sms - previous_modes.at(i).sms) <= 0))
			back_inserted.push_back(i);

		//if state only gives
		else
			class_index_mappings.insert(class_index_mappings.begin(), i);
    
    }

	//append the back inserted elements
	for (size_t i = 0; i < back_inserted.size(); i++)
		class_index_mappings.push_back(back_inserted.at(i));

	return class_index_mappings;
    
}

//this function checks to see if this solution
//would have any cycles in it when the RAG was 
//constructed
bool Scheduler::check_for_cycles(std::vector<int> current_solution){

	//these variables monitor the state of the possible Resource allocation graph
	int CPUs_available = 0;
	int GPUs_available = 0;

	//sort the list into the order that shows safety
	auto sorted_item_indexes = sort_classes(current_solution);

	//process the resources being exclusively returned
	size_t i = 0;
	for (; i < sorted_item_indexes.size(); i++){

		int current_item_index = sorted_item_indexes.at(i);

		if (sorted_item_indexes.at(i) == -1)
			break;

		else{

			CPUs_available += task_table.at(current_item_index).at(current_solution.at(current_item_index)).cores - previous_modes.at(current_item_index).cores;
			GPUs_available += task_table.at(current_item_index).at(current_solution.at(current_item_index)).sms - previous_modes.at(current_item_index).sms;

		}

	}

	//next process potentially unsafe modes
	std::queue<int> unsafe_states;

	for (; i < sorted_item_indexes.size(); i++){

		unsafe_states.push(sorted_item_indexes.at(i));

		if (sorted_item_indexes.at(i) == -1)
			break;

	}

	unsafe_states.push(-1);
	bool popped = false;

	while(!(unsafe_states.size() != 1)){

		auto current = unsafe_states.front();
		unsafe_states.pop();

		if (current == -1 && !popped){

			return false;

		}

		else if (current == -1){

			popped = false;
			unsafe_states.push(-1);

		}

		else {

			int cpus_taken_or_returned = task_table.at(current).at(current_solution.at(current)).cores - previous_modes.at(current).cores;
			int gpus_taken_or_returned = task_table.at(current).at(current_solution.at(current)).sms - previous_modes.at(current).sms;

			if ((CPUs_available > cpus_taken_or_returned && cpus_taken_or_returned > 0) || (GPUs_available > gpus_taken_or_returned && gpus_taken_or_returned > 0)){

				CPUs_available -= cpus_taken_or_returned;
				GPUs_available -= gpus_taken_or_returned;

				popped = true;

			}

			else {

				unsafe_states.push(current);

			}

		}


	}

	//now we are safe, all other tasks are just taking resources, and the 
	//knapsack algorithm ensures we have enough resources for this solution
	return true;
}

//this function builds the RAG for the solution
//that is selected by the knapsack algorithm.
void Scheduler::build_RAG(std::vector<int> current_solution, std::vector<std::vector<vertex>>& final_RAG){

	//sort the list into the order that shows safety
	auto sorted_item_indexes = sort_classes(current_solution);

	//try to construct a DAG that represents the resources passed
	//from one task to another. If we can't, then we can't use this
	//solution.
	std::vector<std::vector<vertex>> RAG;

	//vars
	size_t i = 0;
	bool safe;

	//process all resources being returned
	RAG.push_back(std::vector<vertex>());
	for (; i < sorted_item_indexes.size(); i++){

		int current_item_index = sorted_item_indexes.at(i);

		if (sorted_item_indexes.at(i) == -1){

			i++;
			break;

		}

		else{

			int returned_cpus = task_table.at(current_item_index).at(current_solution.at(current_item_index)).cores - previous_modes.at(current_item_index).cores;
			int returned_gpus = task_table.at(current_item_index).at(current_solution.at(current_item_index)).sms - previous_modes.at(current_item_index).sms;

			RAG.at(0).push_back(vertex(returned_cpus, returned_gpus, current_item_index, std::vector<item_map>()));

		}

	}

	//next process potentially unsafe modes
	int current_state_iteration = 0;
	std::vector<vertex> unsafe_states;

	for (; i < sorted_item_indexes.size(); i++){

		int current_item_index = sorted_item_indexes.at(i);

		if (sorted_item_indexes.at(i) == -1){

			i++;
			break;
			
		}

		else{
			
			//if resources are negative then we need to take resources
			int cpus_taken_or_returned = task_table.at(current_item_index).at(current_solution.at(current_item_index)).cores - previous_modes.at(current_item_index).cores;
			int gpus_taken_or_returned = task_table.at(current_item_index).at(current_solution.at(current_item_index)).sms - previous_modes.at(current_item_index).sms;

			unsafe_states.push_back(vertex(cpus_taken_or_returned, gpus_taken_or_returned, current_item_index, std::vector<item_map>()));
		}

	}

	//while we still have unsafe modes to check, keep checking
	//with the assumption being that if we exacerbate the current
	//layer, we can check the layer we just formed
	while (!unsafe_states.empty()){

		std::vector<int> unsafe_states_to_erase;

		RAG.push_back(std::vector<vertex>());

		//loop over all unsafe states not yet satisifed
		for (size_t i = 0; i < unsafe_states.size(); i++){

			int current_item_index = unsafe_states.at(i).task_id;
				
			//if resources are negative then we need to take resources
			int cpus_taken_or_returned = unsafe_states.at(i).core_A;
			int gpus_taken_or_returned = unsafe_states.at(i).core_B;

			//bool to check if we already satisfied resource handoff
			bool cpus_needed = cpus_taken_or_returned <= 0;
			bool gpus_needed = gpus_taken_or_returned <= 0;

			//object to hold the resource map that we make
			auto resource_map = vertex(0, 0, current_item_index, std::vector<item_map>());

			if (cpus_taken_or_returned > 0)
				resource_map.core_A = cpus_taken_or_returned;

			if (gpus_taken_or_returned > 0)
				resource_map.core_B = gpus_taken_or_returned;

			//now look through first layer to see if we can take resources from any of the tasks
			//that returned them safely
			for (size_t j = 0; j < RAG.at(current_state_iteration).size(); j++){

				int safe_task_returned_cpus = RAG.at(current_state_iteration).at(j).core_A;
				int safe_task_returned_gpus = RAG.at(current_state_iteration).at(j).core_B;

				//if both resources are satisfied break
				if (!cpus_needed && !gpus_needed){
				
					RAG.at(current_state_iteration + 1).push_back(resource_map);
					unsafe_states_to_erase.push_back(i);
					break;

				}

				//if we can take resources from this task
				if (safe_task_returned_cpus >= cpus_taken_or_returned || safe_task_returned_gpus >= gpus_taken_or_returned){

					//first check cpus
					if (cpus_needed){

						if (safe_task_returned_cpus >= 0){

							int resources_being_taken = std::min(safe_task_returned_cpus, cpus_taken_or_returned);

							//if we are taking resources from a task already mapped, then add us to its vector
							RAG.at(current_state_iteration).at(j).children.push_back(item_map({CORE_A, resources_being_taken, current_item_index}));

							//update the resources taken
							RAG.at(current_state_iteration).at(j).core_A -= resources_being_taken;

							//update the resources taken
							cpus_taken_or_returned -= resources_being_taken;

							//if we have taken all the resources we need, break
							if (cpus_taken_or_returned == 0)
								cpus_needed = false;

							else
								unsafe_states.at(i).core_A = cpus_taken_or_returned;

						}

					}

					//next check gpus
					if (gpus_needed){

						if (safe_task_returned_gpus >= 0){

							int resources_being_taken = std::min(safe_task_returned_gpus, gpus_taken_or_returned);

							//if we are taking resources from a task already mapped, then add us to its vector
							RAG.at(current_state_iteration).at(j).children.push_back(item_map({CORE_B, resources_being_taken, current_item_index}));

							//update the resources taken
							RAG.at(current_state_iteration).at(j).core_B -= resources_being_taken;

							//update the resources taken
							gpus_taken_or_returned -= resources_being_taken;

							//if we have taken all the resources we need, break
							if (gpus_taken_or_returned == 0)
								gpus_needed = false;

							else
								unsafe_states.at(i).core_B = gpus_taken_or_returned;

						}

					}

				}

			}

		}

		//if we make it here, the solution is still valid but has a cycle
		if (RAG.at(current_state_iteration + 1).empty()){

			safe = false;
			break;

		}

		//otherwise, erase the indicies indicated by the unsafe states
		//FIXME: THIS IS DANGEROUS AS HELL
		auto unsafe_states_copy = unsafe_states;
		unsafe_states.clear();
		for (size_t i = 0; i < unsafe_states_copy.size(); i++)
			if (std::find(unsafe_states_to_erase.begin(), unsafe_states_to_erase.end(), i) == unsafe_states_to_erase.end())
				unsafe_states.push_back(unsafe_states_copy.at(i));

		//if we did not detect a cycle and the unsafe  states are empty
		//add in the safe states that are only consuming resources
		if (unsafe_states.empty()){

			for (; i < sorted_item_indexes.size(); i++){

				int current_item_index = sorted_item_indexes.at(i);
				
				//if resources are negative then we need to take resources
				int cpus_taken_or_returned = task_table.at(current_item_index).at(current_solution.at(current_item_index)).cores - previous_modes.at(current_item_index).cores;
				int gpus_taken_or_returned = task_table.at(current_item_index).at(current_solution.at(current_item_index)).sms - previous_modes.at(current_item_index).sms;

				unsafe_states.push_back(vertex(cpus_taken_or_returned, gpus_taken_or_returned, current_item_index, std::vector<item_map>()));

			}

		}

	}

	//if we have a cycle, then we need to skip this solution
	if ((!barrier && safe) || barrier)
		final_RAG = RAG;

	return;

}

//Implement scheduling algorithm
//NOTES: GPU LOSS HAS AN ENTRY IN
//THE TABLE (0.0) BUT IS NOT USED IN THE
//ALGORITHM. MIGHT BE ADDED AS A FUTURE CONSTRAINT.
void Scheduler::do_schedule(size_t maxCPU){

	//dynamic programming table
	int N = task_table.size();
    std::vector<std::vector<std::vector<std::pair<double, double>>>> dp(N + 1, std::vector<std::vector<std::pair<double, double>>>(maxCPU + 1, std::vector<std::pair<double, double>>(maxSMS + 1, {100000, 100000})));
    std::map<std::tuple<int,int,int>, std::vector<int>> solutions;

	//try to construct a DAG that represents the resources passed
	//from one task to another. If we can't, then we can't use this
	//solution.
	std::vector<std::vector<vertex>> final_RAG;

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
                dp[i][w][v] = {-1.0, -1.0};  
                
				//if the class we are considering is not allowed to switch modes
				//just treat it as though we did check it normally, and only allow
				//looking at the current mode.
				if (!(schedule.get_task(i - 1))->get_changeable()){

						auto item = task_table.at(i - 1).at((schedule.get_task(i - 1))->get_current_mode());

						//if item fits in both sacks
						if ((w >= item.cores) && (v >= item.sms) && (dp[i - 1][w - item.cores][v - item.sms].first != -1)) {

							double newCPULoss = dp[i - 1][w - item.cores][v - item.sms].first - item.cpuLoss;
							
							//if found solution is better, update
							bool safe = true;
							if ((newCPULoss) > (dp[i][w][v].first)) {

								//check if we are on the final item to add
								//if we are, then we need to check to make 
								//sure that the resulting RAG is acyclical
								if (!barrier && i == num_tasks && !first_time){

									//fetch all the items in the candidate solution
									auto current_solution = solutions[{i - 1, w - item.cores, v - item.sms}];
									current_solution.push_back((schedule.get_task(i - 1))->get_current_mode());

									//check the RAG for safety
									safe = check_for_cycles(current_solution);

									//if there are no cycles in the RAG, then we can use this solution
									if (safe){

										dp[i][w][v] = {newCPULoss, 0.0};

										solutions[{i, w, v}] = solutions[{i - 1, w - item.cores, v - item.sms}];
										solutions[{i, w, v}].push_back((schedule.get_task(i - 1))->get_current_mode());

									}
								}

								else {

									dp[i][w][v] = {newCPULoss, 0.0};

									solutions[{i, w, v}] = solutions[{i - 1, w - item.cores, v - item.sms}];
									solutions[{i, w, v}].push_back((schedule.get_task(i - 1))->get_current_mode());

								}
							}
						}
					}

				else {

					//for each item in class
					for (size_t j = 0; j < task_table.at(i - 1).size(); j++) {

						auto item = task_table.at(i - 1).at(j);

						//if item fits in both sacks
						if ((w >= item.cores) && (v >= item.sms) && (dp[i - 1][w - item.cores][v - item.sms].first != -1)) {

							double newCPULoss = dp[i - 1][w - item.cores][v - item.sms].first - item.cpuLoss;
							
							//if found solution is better, update
							bool safe = true;
							if ((newCPULoss) > (dp[i][w][v].first)) {

								//check if we are on the final item to add
								//if we are, then we need to check to make 
								//sure that the resulting RAG is acyclical
								if (!barrier && i == num_tasks && !first_time){

									//fetch all the items in the candidate solution
									auto current_solution = solutions[{i - 1, w - item.cores, v - item.sms}];
									current_solution.push_back(j);

									//check the RAG for safety
									safe = check_for_cycles(current_solution);

									//if there are no cycles in the RAG, then we can use this solution
									if (safe){

										dp[i][w][v] = {newCPULoss, 0.0};

										solutions[{i, w, v}] = solutions[{i - 1, w - item.cores, v - item.sms}];
										solutions[{i, w, v}].push_back(j);

									}
								}

								else {

									dp[i][w][v] = {newCPULoss, 0.0};

									solutions[{i, w, v}] = solutions[{i - 1, w - item.cores, v - item.sms}];
									solutions[{i, w, v}].push_back(j);

								}
							}
						}
					}
				}
     		}
   		}
 	}

    //return optimal solution
	auto result = solutions[{N, maxCPU, maxSMS}];

	//check to see that we got a solution that renders this system schedulable
	if (result.size() == 0 && previous_modes.empty()){

		print_module::print(std::cerr, "Error: System is not schedulable in any configuration. Exiting.\n");
		killpg(process_group, SIGINT);
		return;

	}
	
	else if (result.size() == 0){

		print_module::print(std::cerr, "Error: System is not schedulable in any configuration with specified constraints. Not updating modes.\n");
		return;

	}

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

		//if it's the first time, then the RAG has no bearing 
		//on the allocations
		for (size_t i = 0; i < result.size(); i++)
			previous_modes.push_back(task_table.at(i).at(result.at(i)));

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
		//(don't worry about holding TPC 1)
		int next_TPC = 0;

		for (int i = 0; i < schedule.count(); i++){

			if ((schedule.get_task(i))->get_current_lowest_GPU() > 0){

				print_module::print(std::cerr, "Error in task ", i, ": all tasks should have had lowest GPU cleared. (this likely means memory was not cleaned up)\n");
				killpg(process_group, SIGKILL);
				return;

			}

			//if this task actually has any TPCs assigned
			if (!(schedule.get_task(i))->pure_cpu_task()){

				(schedule.get_task(i))->set_current_lowest_GPU(next_TPC);
				next_TPC += (schedule.get_task(i))->get_current_GPUs();

				if (next_TPC > (int)(maxSMS) + 1){

					print_module::print(std::cerr, "Error in task ", i, ": too many GPUs have been allocated.", next_TPC, " ", maxSMS, " \n");
					killpg(process_group, SIGKILL);
					return;

				}

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

		//from this point onward we have a RAG which dictates how we delegate resources
		//either we have to partition them in a way that causes no cycles, or we barrier
		//and efficiency of transfer does not matter all that much. Either way, we do not 
		//need the old logic for thread handoff in it's entirety.
		build_RAG(result, final_RAG);

		//by this point the RAG has either the previous solution inside of it, or it has
		//the current solution. Either way, we need to update the previous modes to reflect
		//the current modes.
		for (size_t i = 0; i < result.size(); i++)
			previous_modes.at(i) = task_table.at(i).at(result.at(i));

		//now walk the RAG and see what resources need to be passed
		//from which tasks to which tasks
		for(size_t i = 0; i < final_RAG.size(); i++){

			//process the resources being exclusively returned
			for (size_t j; j < final_RAG.at(i).size(); j++){

				size_t CPUs_given_up = final_RAG.at(i).at(j).core_A * -1;
				size_t GPUs_given_up = final_RAG.at(i).at(j).core_B * -1;

				auto task_owned_cpus = (schedule.get_task(final_RAG.at(i).at(j).task_id))->get_cpu_owned_by_process();
				auto task_owned_gpus = (schedule.get_task(final_RAG.at(i).at(j).task_id))->get_gpu_owned_by_process();

				//let the task know what it should give up when it can change modes
				(schedule.get_task(final_RAG.at(i).at(j).task_id))->set_CPUs_change(CPUs_given_up);
				(schedule.get_task(final_RAG.at(i).at(j).task_id))->set_GPUs_change(GPUs_given_up);

				(schedule.get_task(final_RAG.at(i).at(j).task_id))->set_mode_transition(false);

				//for each child we have, give them the resources and keep track of what this task gave up
				//remember, each task always goes down from their highest core id to the lowest
				for (size_t k = 0; k < final_RAG.at(i).at(j).children.size(); k++){
					
					if (final_RAG.at(i).at(j).children.at(k).resource_type == CORE_A){

						std::vector<int> cpus_being_given;

						for (int z = 0; z < final_RAG.at(i).at(j).children.at(k).resource_amount; z++){

							cpus_being_given.push_back(task_owned_cpus.at(task_owned_cpus.size() - 1));
							task_owned_cpus.pop_back();

						}

						(schedule.get_task(final_RAG.at(i).at(j).children.at(k).task_id))->set_cpus_granted_from_other_tasks({final_RAG.at(i).at(j).task_id, cpus_being_given});

					}

					if (final_RAG.at(i).at(j).children.at(k).resource_type == CORE_B){

						std::vector<int> gpus_being_given;

						for (int z = 0; z < final_RAG.at(i).at(j).children.at(k).resource_amount; z++){

							gpus_being_given.push_back(task_owned_gpus.at(task_owned_gpus.size() - 1));
							task_owned_gpus.pop_back();

						}

						(schedule.get_task(final_RAG.at(i).at(j).children.at(k).task_id))->set_gpus_granted_from_other_tasks({final_RAG.at(i).at(j).task_id, gpus_being_given});

					}

				}

			}

		}

		/*//CPU first
		for (int i = 0; i < schedule.count(); i++){
		
			int gained = (schedule.get_task(i))->get_current_CPUs() - (schedule.get_task(i))->get_previous_CPUs(); 
			(schedule.get_task(i))->set_CPUs_gained(gained);

		}

		//GPU next
		for (int i = 0; i < schedule.count(); i++){
		
			int gained = (schedule.get_task(i))->get_current_GPUs() - (schedule.get_task(i))->get_previous_GPUs(); 
			(schedule.get_task(i))->set_GPUs_gained(gained);

		}

		//honestly I see no need to determine which GPUs get passed for now
		//this will become useful when we want to allow GPC alignment, but
		//for now just take the highest TPC and give it to a pool of TPCs that
		//are not assigned. Then just assign the TPCs in the pool to the tasks 
		//that should be gaining
		std::vector<int> TPC_pool;
		for (int i = 0; i < schedule.count(); i++){

			//if we are supposed to be giving GPUs
			if ((schedule.get_task(i))->get_GPUs_gained() < 0){

				auto TPCs = (schedule.get_task(i))->retract_GPUs((schedule.get_task(i))->get_GPUs_gained());

				for (size_t j = 0; j < TPCs.size(); j++)
					TPC_pool.push_back(TPCs.at(j));
			}

		}

		for (int i = 0; i < schedule.count(); i++){

			//if we are supposed to be gaining GPUs
			if ((schedule.get_task(i))->get_GPUs_gained() > 0){

				std::vector<int> TPCS_granted;

				for (int j = 0; j < (schedule.get_task(i))->get_GPUs_gained(); j++){

					TPCS_granted.push_back(TPC_pool.at(0));
					TPC_pool.erase(TPC_pool.begin());

				}

				(schedule.get_task(i))->gifted_GPUs(TPCS_granted);

			}

		}

		//First determine how many CPUs each task gets from each other task.
		for (int i = 0; i < schedule.count(); i++){

			for (int j = i + 1; j < schedule.count(); j++){

				int task_gaining_cpus = -1;
				int task_giving_cpus = -1;

				//if task "j" is supposed to be giving CPUs
				if ((schedule.get_task(i))->get_CPUs_gained() > 0 && (schedule.get_task(j))->get_CPUs_gained() < 0){
					
					//comment with code
					task_giving_cpus = j;
					task_gaining_cpus = i;

				}

				//if task "i" is supposed to be giving CPUs
				else if ((schedule.get_task(j))->get_CPUs_gained() > 0 && (schedule.get_task(i))->get_CPUs_gained() < 0){

					//comment with code
					task_giving_cpus = i;
					task_gaining_cpus = j;

				}

				//this condition can be false if both tasks are gaining CPUs or both are losing CPUs
				if (task_gaining_cpus != -1 && task_giving_cpus != -1){

					int difference = abs((schedule.get_task(task_gaining_cpus))->get_CPUs_gained()) - abs((schedule.get_task(task_giving_cpus))->get_CPUs_gained());
					
					//if positive, then task i is gaining CPUs, if negative, task j is gaining CPUs
					int amount = (difference >= 0) ? abs((schedule.get_task(task_giving_cpus))->get_CPUs_gained()) : abs((schedule.get_task(task_gaining_cpus))->get_CPUs_gained());

					(schedule.get_task(task_gaining_cpus))->set_CPUs_gained((schedule.get_task(task_gaining_cpus))->get_CPUs_gained() - amount);
					(schedule.get_task(task_giving_cpus))->set_CPUs_gained((schedule.get_task(task_giving_cpus))->get_CPUs_gained() + amount);
					(schedule.get_task(task_gaining_cpus))->update_give(task_giving_cpus, -1 * amount);
					(schedule.get_task(task_giving_cpus))->update_give(task_gaining_cpus, amount);
					
				}
			}
		}

		//Now determine how many GPUs each task gets from each other task.
		for (int i = 0; i < schedule.count(); i++){

			for (int j = i + 1; j < schedule.count(); j++){

				int task_gaining_gpus = -1;
				int task_giving_gpus = -1;

				//if task "j" is supposed to be giving CPUs
				if ((schedule.get_task(i))->get_GPUs_gained() > 0 && (schedule.get_task(j))->get_GPUs_gained() < 0){
					
					//comment with code
					task_giving_gpus = j;
					task_gaining_gpus = i;

				}

				//if task "i" is supposed to be giving CPUs
				else if ((schedule.get_task(j))->get_GPUs_gained() > 0 && (schedule.get_task(i))->get_GPUs_gained() < 0){

					//comment with code
					task_giving_gpus = i;
					task_gaining_gpus = j;

				}

				//this condition can be false if both tasks are gaining CPUs or both are losing CPUs
				if (task_gaining_gpus != -1 && task_giving_gpus != -1){

					int difference = abs((schedule.get_task(task_gaining_gpus))->get_GPUs_gained()) - abs((schedule.get_task(task_giving_gpus))->get_GPUs_gained());
					
					//if positive, then task i is gaining CPUs, if negative, task j is gaining CPUs
					int amount = (difference >= 0) ? abs((schedule.get_task(task_giving_gpus))->get_GPUs_gained()) : abs((schedule.get_task(task_gaining_gpus))->get_GPUs_gained());

					(schedule.get_task(task_gaining_gpus))->set_GPUs_gained((schedule.get_task(task_gaining_gpus))->get_GPUs_gained() - amount);
					(schedule.get_task(task_giving_gpus))->set_GPUs_gained((schedule.get_task(task_giving_gpus))->get_GPUs_gained() + amount);
					(schedule.get_task(task_gaining_gpus))->update_gpu_give(task_giving_gpus, -1 * amount);
					(schedule.get_task(task_giving_gpus))->update_gpu_give(task_gaining_gpus, amount);
					
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
					//NOTE: give preference to CPUs that are active in the giving task
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
					//NOTES: if we have to continue giving CPUs then give up some of our passive CPUs
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
		}*/
	}

	first_time = false;

}

void Scheduler::setTermination(){
	schedule.setTermination();
}