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

		//the loss function is different if the 
		//task is a pure cpu task or hybrid task
		if (taskData_object->pure_cpu_task())
			item.cpuLoss = (1.0 / taskData_object->get_elasticity() * (std::pow(taskData_object->get_max_utilization() - (taskData_object->get_work(j) / taskData_object->get_period(j)), 2)));
		
		else 
			item.cpuLoss = (1.0 / taskData_object->get_elasticity() * (std::pow(taskData_object->get_max_utilization() - ((taskData_object->get_work(j) / taskData_object->get_period(j)) + (taskData_object->get_GPU_work(j) / taskData_object->get_period(j))), 2)));

		item.gpuLoss = 0;
		item.cores = taskData_object->get_CPUs(j);
		item.sms = taskData_object->get_GPUs(j);

		task_table.at(task_table.size() - 1).push_back(item);
		
	}

	return taskData_object;
}

//static vector sizes
std::vector<std::vector<Scheduler::task_mode>> Scheduler::task_table(100, std::vector<task_mode>(100));

//helper function for checking for cycles in RAG
bool Scheduler::has_cycle(const std::unordered_map<int, Node>& nodes, int start) {

    std::unordered_set<int> visited;
    std::unordered_set<int> recursion_stack;
    
    std::function<bool(int)> dfs = [&](int node_id) -> bool {

        visited.insert(node_id);
        recursion_stack.insert(node_id);
        
        for (const Edge& edge : nodes.at(node_id).edges) {

            if (!visited.count(edge.to_node)){
                if (dfs(edge.to_node)) 
					return true;
			}
            
            else if (recursion_stack.count(edge.to_node))
                return true;

        }
        
        recursion_stack.erase(node_id);
        return false;

    };
    
    return dfs(start);
}

//Returns true if graph was successfully built, false if impossible due to cycles
bool Scheduler::build_resource_graph(std::vector<std::pair<int, int>> resource_pairs, 
                        std::unordered_map<int, Node>& nodes, std::unordered_map<int, Node>& static_nodes) {
    nodes.clear();
    
    //create all nodes
    for (size_t i = 0; i < resource_pairs.size(); i++) {

		auto [x, y] = resource_pairs[i];
		nodes[i] = Node{(int)i, x, y, {}};
		static_nodes[i] = nodes[i];
    
	}
    
    //Sort nodes by resource availability (safe first)
    std::vector<int> provider_order;
    std::vector<int> consumer_order;
    
    for (const auto& [id, node] : nodes) {
    
	    if (node.x > 0 || node.y > 0)
            provider_order.push_back(id);
    
	    if (node.x < 0 || node.y < 0)
            consumer_order.push_back(id);

    }
    
    //Try to satisfy each consumer's needs
    for (int consumer_id : consumer_order) {

        Node& consumer = nodes[consumer_id];
        int needed_x = -consumer.x;
        int needed_y = -consumer.y;
        
        for (int provider_id : provider_order) {

            if (provider_id == consumer_id) continue;
            
            Node& provider = nodes[provider_id];
            Edge new_edge{consumer_id, 0, 0};
            bool edge_needed = false;
            
            //Try to satisfy x resource need
            if (needed_x > 0 && provider.x > 0) {

                int transfer = std::min(needed_x, provider.x);
                new_edge.x_amount = transfer;
                needed_x -= transfer;
                edge_needed = true;

            }
            
            //Try to satisfy y resource need
            if (needed_y > 0 && provider.y > 0) {

                int transfer = std::min(needed_y, provider.y);
                new_edge.y_amount = transfer;
                needed_y -= transfer;
                edge_needed = true;

            }
            
            //If this edge would transfer resources, add it and check for cycles
            if (edge_needed) {
                
				provider.edges.push_back(new_edge);

                if (has_cycle(nodes, provider_id))
                    provider.edges.pop_back();
				
				else {

                    //Update provider's available resources
                    provider.x -= new_edge.x_amount;
                    provider.y -= new_edge.y_amount;
                
				}

            }

        }
        
        //Check if all needs were satisfied
        if (needed_x > 0 || needed_y > 0)
            return false;
    }
    
    return true;

}

//convert the print_graph function to use buffered print
void Scheduler::print_graph(const std::unordered_map<int, Node>& nodes, std::unordered_map<int, Node> static_nodes) {

	std::ostringstream mode_strings;

	print_module::buffered_print(mode_strings, "\nNodes and resource transfers:\n");
	
	for (const auto& [id, node] : nodes) {

		if (id != ((int) nodes.size() - 1))
			print_module::buffered_print(mode_strings, "Node ", id, " <", static_nodes[id].x, ",", static_nodes[id].y, "> → ");
		else
			print_module::buffered_print(mode_strings, "Free Resources", " <", static_nodes[id].x, ",", static_nodes[id].y, "> → ");

		if (node.edges.empty())
			print_module::buffered_print(mode_strings, "no edges");

		else {
			
			for (const Edge& edge : node.edges) {

				print_module::buffered_print(mode_strings, edge.to_node, "(");
				bool first = true;

				if (edge.x_amount > 0) {
				
				    print_module::buffered_print(mode_strings, "x:", edge.x_amount);
					first = false;
				
				}

				if (edge.y_amount > 0) {
				
				    if (!first) print_module::buffered_print(mode_strings, ",");
					print_module::buffered_print(mode_strings, "y:", edge.y_amount);
				
				}
				
				print_module::buffered_print(mode_strings, ") ");
			}
		}
		print_module::buffered_print(mode_strings, "\n");
	}
	print_module::buffered_print(mode_strings, "\n");

	print_module::flush(std::cerr, mode_strings);
}

//Implement scheduling algorithm
void Scheduler::do_schedule(size_t maxCPU){

	//vector for transitioned tasks
	std::vector<int> transitioned_tasks;

	//for each run we need to see what resources are left in the pool from the start
	int starting_CPUs = NUMCPUS - 1;
	int starting_GPUs = maxSMS;

	if (!first_time){

		for (int i = 0; i < schedule.count(); i++){

			starting_CPUs -= previous_modes.at(i).cores;
			starting_GPUs -= previous_modes.at(i).sms;

		}

	}

	else {

		//add an entry for each task into previous modes
		for (int i = 0; i < schedule.count(); i++)
			previous_modes.push_back(task_mode());

	}

	//dynamic programming table
	int N = task_table.size();
    std::vector<std::vector<std::vector<std::pair<double, std::pair<int, int>>>>> dp(N + 1, std::vector<std::vector<std::pair<double, std::pair<int, int>>>>(maxCPU + 1, std::vector<std::pair<double, std::pair<int, int>>>(maxSMS + 1, {100000, {starting_CPUs, starting_GPUs}})));
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
                dp[i][w][v] = {-1.0, {0, 0}};  
                
				//if the class we are considering is not allowed to switch modes
				//just treat it as though we did check it normally, and only allow
				//looking at the current mode.
				if (!(schedule.get_task(i - 1))->get_changeable()){

						//fetch item definition
						auto item = task_table.at(i - 1).at((schedule.get_task(i - 1))->get_current_mode());

						//fetch initial suspected resource values
						size_t current_item_sms = item.sms;
						size_t current_item_cores = item.cores;

						//if this item is feasible at all 
						if ((w >= current_item_cores) && (v >= current_item_sms) && (dp[i - 1][w - current_item_cores][v - current_item_sms].first != -1)){

							//fetch the current resource pool
							auto current_resource_pool = dp[i - 1][w - current_item_cores][v - current_item_sms].second;

							//if we have no explicit sync,
							//then we have to do safety checks
							if (!barrier && !first_time){

								//fetch the guaranteed resources
								int returned_cpus = current_resource_pool.first;
								int returned_gpus = current_resource_pool.second;
								
								//negative means we are returning resources
								int cpu_change = item.cores - previous_modes.at(i - 1).cores;
								int sm_change = item.sms - previous_modes.at(i - 1).sms;

								//check if this mode could cause a cycle
								if ((cpu_change < 0 && sm_change > 0) || (sm_change < 0 && cpu_change > 0)){
									
									returned_cpus -= cpu_change;
									returned_gpus -= sm_change;

									//we have to change our resource demands to make
									//the system safe again
									if (returned_cpus < 0)
										current_item_cores = std::max(item.cores, previous_modes.at(i - 1).cores);

									if (returned_gpus < 0)
										current_item_sms = std::max(item.sms, previous_modes.at(i - 1).sms);

								}
							}

							//update the current resource pool
							current_resource_pool = dp[i - 1][w - current_item_cores][v - current_item_sms].second;

							//if item fits in both sacks
							if ((w >= current_item_cores) && (v >= current_item_sms) && (dp[i - 1][w - current_item_cores][v - current_item_sms].first != -1)) {

								double newCPULoss = dp[i - 1][w - current_item_cores][v - current_item_sms].first - item.cpuLoss;

								//update the pool at this stage
								current_resource_pool.first -= (current_item_cores - previous_modes.at(i - 1).cores);
								current_resource_pool.second -= (current_item_sms - previous_modes.at(i - 1).sms);
								
								//if found solution is better, update
								if ((newCPULoss) > (dp[i][w][v].first)) {

										dp[i][w][v] = {newCPULoss, current_resource_pool};

										solutions[{i, w, v}] = solutions[{i - 1, w - current_item_cores, v - current_item_sms}];
										solutions[{i, w, v}].push_back((schedule.get_task(i - 1))->get_current_mode());
									
								}
							}
						}
					}

				else {

					//for each item in class
					for (size_t j = 0; j < task_table.at(i - 1).size(); j++) {

						auto item = task_table.at(i - 1).at(j);

						//fetch initial suspected resource values
						size_t current_item_sms = item.sms;
						size_t current_item_cores = item.cores;

						//if this item is feasible at all 
						if ((w >= current_item_cores) && (v >= current_item_sms) && (dp[i - 1][w - current_item_cores][v - current_item_sms].first != -1)){

							//fetch the current resource pool
							auto current_resource_pool = dp[i - 1][w - current_item_cores][v - current_item_sms].second;

							//if we have no explicit sync,
							//then we have to do safety checks
							if (!barrier && !first_time){

								//fetch the guaranteed resources
								int returned_cpus = current_resource_pool.first;
								int returned_gpus = current_resource_pool.second;
								
								//negative means we are returning resources
								int cpu_change = item.cores - previous_modes.at(i - 1).cores;
								int sm_change = item.sms - previous_modes.at(i - 1).sms;

								//check if this mode could cause a cycle
								if ((cpu_change < 0 && sm_change > 0) || (sm_change < 0 && cpu_change > 0)){
									
									returned_cpus -= cpu_change;
									returned_gpus -= sm_change;

									//we have to change our resource demands to make
									//the system safe again
									if (returned_cpus < 0){

										current_item_cores = std::max(item.cores, previous_modes.at(i - 1).cores);

									}

									if (returned_gpus < 0){
										
										current_item_sms = std::max(item.sms, previous_modes.at(i - 1).sms);
									
									}

								}
							}

							//update the current resource pool
							current_resource_pool = dp[i - 1][w - current_item_cores][v - current_item_sms].second;

							//if item fits in both sacks
							if ((w >= current_item_cores) && (v >= current_item_sms) && (dp[i - 1][w - current_item_cores][v - current_item_sms].first != -1)) {

								double newCPULoss = dp[i - 1][w - current_item_cores][v - current_item_sms].first - item.cpuLoss;

								//update the pool at this stage
								current_resource_pool.first -= (current_item_cores - previous_modes.at(i - 1).cores);
								current_resource_pool.second -= (current_item_sms - previous_modes.at(i - 1).sms);
								
								//if found solution is better, update
								if ((newCPULoss) > (dp[i][w][v].first)) {

									dp[i][w][v] = {newCPULoss, current_resource_pool};

									solutions[{i, w, v}] = solutions[{i - 1, w - current_item_cores, v - current_item_sms}];
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

		//update the previous modes to the first ever selected modes
		for (size_t i = 0; i < result.size(); i++)
			previous_modes.at(i) = (task_table.at(i).at(result.at(i)));

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

		//assign all the unassigned cpus to the scheduler to hold
		for (int i = next_CPU; i < num_CPUs + 1; i++)
			free_cores_A.push_back(i);

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

				for (int j = 0; j < (schedule.get_task(i))->get_current_GPUs(); j++)
					(schedule.get_task(i))->push_back_gpu(next_TPC ++);

				if (next_TPC > (int)(maxSMS) + 1){

					print_module::print(std::cerr, "Error in task ", i, ": too many GPUs have been allocated.", next_TPC, " ", maxSMS, " \n");
					killpg(process_group, SIGKILL);
					return;

				}

			}
		}

		//assign all the unassigned gpus to the scheduler to hold
		for (int i = next_TPC; i < (int)(maxSMS); i++)
			free_cores_B.push_back(i);

	}

	//Transfer as efficiently as possible.
	//This portion of code is supposed to allocate
	//CPUs from tasks that are not active to ones that
	//should be rescheduling right now... because of this,
	//it should also be a good candidate algorithm for
	//passing off the GPU SMs.
	else {

		//for each mode in result, subtract the new mode from the old mode to determine how many resources are being given up or taken
		//from each task. This will be used to build the RAG.
		std::unordered_map<int, Node> nodes;
		std::unordered_map<int, Node> static_nodes;
		std::vector<std::pair<int, int>> dependencies;

		for (size_t i = 0; i < result.size(); i++){

			//fetch the current mode
			auto current_mode = task_table.at(i).at(result.at(i));

			//fetch the previous mode
			auto previous_mode = previous_modes.at(i);

			//add the new node
			dependencies.push_back({previous_mode.cores - current_mode.cores, previous_mode.sms - current_mode.sms});

		}

		//for all the free cores of both types, add them to the RAG
		//via adding a node that gives up that many resources
		dependencies.push_back({free_cores_A.size(), free_cores_B.size()});

		//if this returns false, then we have a cycle and only a barrier
		//can allow the handoff
		if (build_resource_graph(dependencies, nodes, static_nodes)){

			//show the resource graph (debugging)
			print_module::print(std::cerr, "\n========================= \n", "New Schedule RAG:\n");
	        print_graph(nodes, static_nodes);
			print_module::print(std::cerr, "========================= \n\n");

			//by this point the RAG has either the previous solution inside of it, or it has
			//the current solution. Either way, we need to update the previous modes to reflect
			//the current modes.
			for (size_t i = 0; i < result.size(); i++){

				(schedule.get_task(i))->clear_cpus_granted_from_other_tasks();
				(schedule.get_task(i))->clear_gpus_granted_from_other_tasks();

			}

			//now walk the RAG and see what resources need to be passed
			//from which tasks to which tasks
			for (const auto& [id, node] : nodes) {

				int CPUs_given_up = 0;
				int GPUs_given_up = 0;

				std::vector<int> task_owned_gpus;
				std::vector<int> task_owned_cpus;

				//fetch the current mode
				Scheduler::task_mode current_mode;

				//fetch the previous mode
				Scheduler::task_mode previous_mode;

				//check if the resources are coming from the free pool
				if (id != ((int) nodes.size() - 1)){

					task_owned_gpus = (schedule.get_task(id))->get_gpu_owned_by_process();
					task_owned_cpus = (schedule.get_task(id))->get_cpu_owned_by_process();

					current_mode = task_table.at(id).at(result.at(id));
					previous_mode = previous_modes.at(id);

				}
				
				//if only receiving resources, just skip
				if (node.edges.empty() && id != ((int) nodes.size() - 1)){

					//check that they are not just giving up resources
					//to the free pool
					bool transitioned = false;

					if ((previous_mode.cores - current_mode.cores) > 0){

						for (int i = 0; i < (previous_mode.cores - current_mode.cores); i++){

							free_cores_A.push_back(task_owned_cpus.at(task_owned_cpus.size() - 1));
							task_owned_cpus.pop_back();
						
						}

						transitioned = true;

					}

					if ((previous_mode.sms - current_mode.sms) > 0){

						for (int i = 0; i < (previous_mode.sms - current_mode.sms); i++){

							free_cores_B.push_back(task_owned_gpus.at(task_owned_gpus.size() - 1));
							task_owned_gpus.pop_back();
						
						}

						transitioned = true;

					}

					//let the task know what it should give up when it can change modes
					if (transitioned){

						(schedule.get_task(id))->set_CPUs_change(CPUs_given_up);
						(schedule.get_task(id))->set_GPUs_change(GPUs_given_up);
						(schedule.get_task(id))->set_mode_transition(false);

					}

					continue;

				}
				
				else {

					for (const Edge& edge : node.edges) {

						int task_being_given_to = edge.to_node;

						//if resource type A
						if (edge.x_amount > 0) {

							std::vector<int> cpus_being_given;

							if (id != ((int) nodes.size() - 1) && (int) task_owned_cpus.size() < edge.x_amount){

								print_module::print(std::cerr, "Error: not enough CPUs to give to task ", task_being_given_to, " from task ", id, " size gotten: ", task_owned_cpus.size(), " expected: ", edge.x_amount, ". Exiting.\n");
								killpg(process_group, SIGINT);
								return;

							}

							else if (id == ((int) nodes.size() - 1) && (int) free_cores_A.size() < edge.x_amount){

								print_module::print(std::cerr, "Error: not enough CPUs to give to task ", task_being_given_to, " from free pool. size gotten: ", free_cores_A.size(), " expected: ", edge.x_amount, ". Exiting.\n");
								killpg(process_group, SIGINT);
								return;

							}

							for (int z = 0; z < edge.x_amount; z++){

								if (id != ((int) nodes.size() - 1)){

									cpus_being_given.push_back(task_owned_cpus.at(task_owned_cpus.size() - 1));
									task_owned_cpus.pop_back();
								
								}

								else{

									cpus_being_given.push_back(free_cores_A.at(free_cores_A.size() - 1));
									free_cores_A.pop_back();

								}

							}

							if (id == ((int) nodes.size() - 1))
								(schedule.get_task(task_being_given_to))->set_cpus_granted_from_other_tasks({MAXTASKS, cpus_being_given});
							else
								(schedule.get_task(task_being_given_to))->set_cpus_granted_from_other_tasks({id, cpus_being_given});

							//update other task's cpu change
							auto change_amount = (schedule.get_task(task_being_given_to))->get_CPUs_change();
							(schedule.get_task(task_being_given_to))->set_CPUs_change(change_amount - edge.x_amount);

							//update our own
							CPUs_given_up += edge.x_amount;

						}

						//if resource type B
						if (edge.y_amount > 0) {

							std::vector<int> gpus_being_given;

							if (id != ((int) nodes.size() - 1) && (int) task_owned_gpus.size() < edge.y_amount){

								print_module::print(std::cerr, "Error: not enough GPUs to give to task ", task_being_given_to, " from task ", id, " size gotten: ", task_owned_gpus.size(), " expected: ", edge.y_amount, ". Exiting.\n");
								killpg(process_group, SIGINT);
								return;

							}

							else if (id == ((int) nodes.size() - 1) && (int) free_cores_B.size() < edge.y_amount){

								print_module::print(std::cerr, "Error: not enough GPUs to give to task ", task_being_given_to, " from free pool. size gotten: ", free_cores_B.size(), " expected: ", edge.y_amount, ". Exiting.\n");
								killpg(process_group, SIGINT);
								return;

							}

							for (int z = 0; z < edge.y_amount; z++){

								if (id != ((int) nodes.size() - 1)){

									gpus_being_given.push_back(task_owned_gpus.at(task_owned_gpus.size() - 1));
									task_owned_gpus.pop_back();

								}

								else{

									gpus_being_given.push_back(free_cores_B.at(free_cores_B.size() - 1));
									free_cores_B.pop_back();

								}

							}
							
							if (id == ((int) nodes.size() - 1))
								(schedule.get_task(task_being_given_to))->set_gpus_granted_from_other_tasks({MAXTASKS, gpus_being_given});
							else
								(schedule.get_task(task_being_given_to))->set_gpus_granted_from_other_tasks({id, gpus_being_given});
							

							//update other task's gpu change
							auto change_amount = (schedule.get_task(task_being_given_to))->get_GPUs_change();
							(schedule.get_task(task_being_given_to))->set_GPUs_change(change_amount - edge.y_amount);

							//update our own
							GPUs_given_up += edge.y_amount;

						}

					}

				}

				//check if we gave up resource AND we are giving resources to the free pool
				if ((previous_mode.cores - current_mode.cores) != CPUs_given_up){

					for (int i = CPUs_given_up; i < (previous_mode.cores - current_mode.cores); i++){

						free_cores_A.push_back(task_owned_cpus.at(task_owned_cpus.size() - 1));
						task_owned_cpus.pop_back();

						CPUs_given_up++;
						
					}

				}

				if ((previous_mode.sms - current_mode.sms) != GPUs_given_up){

					for (int i = GPUs_given_up; i < (previous_mode.sms - current_mode.sms); i++){

						free_cores_B.push_back(task_owned_gpus.at(task_owned_gpus.size() - 1));

						task_owned_gpus.pop_back();

						GPUs_given_up++;
						
					}

				}
				
				//let the task know what it should give up when it can change modes
				(schedule.get_task(id))->set_CPUs_change(CPUs_given_up);
				(schedule.get_task(id))->set_GPUs_change(GPUs_given_up);
				(schedule.get_task(id))->set_mode_transition(false);

				//add the task to list of tasks that had to transition
				transitioned_tasks.push_back(id);

			}
		}	
 
	}

	//update the previous modes to the current modes
	for (size_t i = 0; i < result.size(); i++){

		previous_modes.at(i) = task_table.at(i).at(result.at(i));

	}

	//we prevent the scheduler from doing another reschedule until the tasks have
	//actually transitioned
	//for (int id : transitioned_tasks)
	//	while ((schedule.get_task(id))->check_mode_transition() == false);

	first_time = false;
}

void Scheduler::setTermination(){
	schedule.setTermination();
}