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
#include <cstring>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <stack>

#ifdef __NVCC__

	#include <cuda.h>
	#include <cuda_runtime.h>

	#define CUDA_SAFE_CALL(x)                                         \
	do {                                                              \
	CUresult result = x;                                           \
	if (result != CUDA_SUCCESS) {                                  \
		const char *msg;                                            \
		cuGetErrorName(result, &msg);                               \
		std::cerr << "\nerror: " #x " failed with error "           \
					<< msg << '\n';                                   \
		exit(1);                                                    \
	}                                                              \
	} while(0)

	#define CUDA_NEW_SAFE_CALL(x)                                     \
	do {                                                              \
	cudaError_t result = x;                                        \
	if (result != cudaSuccess) {                                   \
		std::cerr << "\nerror: " #x " failed with error "           \
					<< cudaGetErrorName(result) << '\n';              \
		exit(1);                                                    \
	}                                                              \
	} while(0)

	__device__ volatile	double dp_two[25 + 1][128 + 1][128 + 1][3];
	__device__ volatile int solutions[25][128 + 1][128 + 1][2];

	__constant__ int constant_task_table[25 * 8 * 2];

	__constant__ double constant_losses[25 * 8];

	__global__ void set_dp_table(){
		
		for (int i = 0; i < 25 + 1; i++)
			for (int j = 0; j < 128 + 1; j++)
				for (int k = 0; k < 128 + 1; k++)
					for (int l = 0; l < 3; l++)
						dp_two[i][j][k][l] = 100000;

	}

	__global__ void device_do_schedule(int num_tasks, int maxCPU, int NUMGPUS, int* task_table, double* losses, double* final_loss, int* uncooperative_tasks, int* final_solution){

		//loop over all tasks
		for (int i = 1; i <= (int) num_tasks; i++) {

			//gather task info
			int j_start = 0;
			int j_end = 8;

			//check if cooperative
			if ((uncooperative_tasks[i - 1])){

				j_start = uncooperative_tasks[i - 1];
				j_end = j_start + 1;

			}

			//assume 1 block of 1024 threads for now
			int pass_count = ceil(((maxCPU + 1) * (NUMGPUS + 1)) / 1024) + 1;
			for (int k = 0; k < pass_count; k++){

				//w = cpu
				int w = (((k * 1024) + threadIdx.x) / (NUMGPUS + 1));
				
				//v = gpu
				int v = (((k * 1024) + threadIdx.x) % (NUMGPUS + 1));

				//invalid state
				dp_two[i][w][v][0] = -1.0;

				//for each item in class
				for (size_t j = j_start; j < j_end; j++) {

					//fetch initial suspected resource values
					int current_item_sms = constant_task_table[(i - 1) * 8 * 2 + j * 2 + 1];
					int current_item_cores = constant_task_table[(i - 1) * 8 * 2 + j * 2];

					if (current_item_cores == -1 || current_item_sms == -1)
						continue;

					//if item fits in both sacks
					if ((w >= current_item_cores) && (v >= current_item_sms) && (dp_two[i - 1][w - current_item_cores][v - current_item_sms][0] != -1)) {

						double newCPULoss_two = dp_two[i - 1][w - current_item_cores][v - current_item_sms][0] - constant_losses[(i - 1) * 8 + j];
						
						//if found solution is better, update
						if ((newCPULoss_two) > (dp_two[i][w][v][0])) {

							dp_two[i][w][v][0] = newCPULoss_two;

							//store j into the corresponding slot of the 1d array in the first position
							solutions[i][w][v][0] = j;

							//store a pointer to the previous portion of the solution in the second position
							solutions[i][w][v][1] = ((char)(i - 1) << 16) | ((char)(w - current_item_cores) << 8) | (char)(v - current_item_sms);

						}

					}

				}

				__syncthreads();

			}

		}

		//to get the final answer, start at the end and work backwards, taking the j values
		if (threadIdx.x < 1){

			int pivot = solutions[num_tasks][maxCPU][NUMGPUS][1];
			
			final_solution[num_tasks - 1] = solutions[num_tasks][maxCPU][NUMGPUS][0];

			for (int i = num_tasks - 2; i >= 0; i--){

				//im lazy and bad at math apparently
				int first = (pivot >> 16) & 0xFF;
				int second = (pivot >> 8) & 0xFF;
				int third = pivot & 0xFF;

				pivot = solutions[first][second][third][1];
				final_solution[i] = solutions[first][second][third][0];

			}

			//print the final loss 
			*final_loss = 100000 - dp_two[num_tasks][maxCPU][NUMGPUS][0];

		}

	}

#endif


class Schedule * Scheduler::get_schedule(){
	return &schedule;
}

int Scheduler::get_num_tasks(){
	return task_table.size();
}

void Scheduler::generate_unsafe_combinations(size_t maxCPU){

	std::vector<int> unsafe_tasks;

	//find the unsafe tasks
	for (int i = 0; i < (int) task_table.size(); i++){

		//for each mode
		for (int j = 0; j < (int) task_table.at(i).size(); j++){

			//if this mode is unsafe
			if (task_table.at(i).at(j).unsafe_mode){

				//add to the unsafe tasks
				unsafe_tasks.push_back(i);
				break;

			}

		}

	}

	//temporary tables
	double dp_two[unsafe_tasks.size() + 1][maxCPU + 1][NUMGPUS + 1][3];
	int solutions[num_tasks + 1][maxCPU + 1][NUMGPUS + 1][num_tasks];

	//run the knapsack algorithm
	for (int i = 1; i <= (int) unsafe_tasks.size(); i++) {

		//only use the unsafe tasks
		int z = unsafe_tasks.at(i - 1);

		for (int w = 0; w <= (int) maxCPU; w++) {

			for (int v = 0; v <= (int) NUMGPUS; v++) {

				//invalid state
				dp_two[i][w][v][0] = -1.0;
				dp_two[i][w][v][1] = 0.0;
				dp_two[i][w][v][2] = 0.0;

				//for each item in class
				for (size_t j = 0; j < task_table.at(z).size(); j++) {

					auto item = task_table.at(z).at(j);

					//fetch initial suspected resource values
					int current_item_sms = item.sms;
					int current_item_cores = item.cores;

					//if this item is feasible at all 
					if ((w >= current_item_cores) && (v >= current_item_sms) && (dp_two[i - 1][w - current_item_cores][v - current_item_sms][0] != -1)){
						
						//all values are static at value of 1
						double newCPULoss_two = dp_two[i - 1][w - current_item_cores][v - current_item_sms][0] - 1;
						
						//no solution will ever be "better" when at pos - 1, but
						//we will fill out all possible combinations in the table
						if ((newCPULoss_two) > (dp_two[i][w][v][0])) {

							dp_two[i][w][v][0] = newCPULoss_two;

							std::memcpy(solutions[i][w][v], solutions[i - 1][w - current_item_cores][v - current_item_sms], num_tasks * sizeof(int));

							solutions[i][w][v][i - 1] = j;

						}

					}

				}

			}

		}

	}

	//malloc the unsafe table
	unsafe_table = (int*) malloc(sizeof(int) * (maxCPU + 1 + NUMGPUS + 1 + num_tasks));

	//after this, we have all possible unsafe combinations in the last
	//layer. We need to fill out all empty slots now with answers which
	//are at least as good
	int i = (int) unsafe_tasks.size();

	for (int w = (int) maxCPU; w > 0; w--) {

		int last_largest_v = 0;

		for (int v = (int) NUMGPUS; v > 0; v--) {

			//if we have an invalid state
			if (dp_two[i][w][v][0] != -1.0)
				last_largest_v = v;

			//store in the permanent table
			std::memcpy(&unsafe_table[w * ((NUMGPUS + 1) * (num_tasks)) + v * (num_tasks)], solutions[i][w][last_largest_v], num_tasks * sizeof(int));

		}

	}

	//check from the other direction now
	for (int v = (int) NUMGPUS; v > 0; v--) {

		int last_largest_w = 0;

		for (int w = (int) maxCPU; w > 0; w--) {

			//if we have an invalid state
			if (dp_two[i][w][v][0] != -1.0)
				last_largest_w = w;

			//store in the permanent table
			std::memcpy(&unsafe_table[w * ((NUMGPUS + 1) * (num_tasks)) + v * (num_tasks)], solutions[i][last_largest_w][v], num_tasks * sizeof(int));

		}

	}

}

TaskData * Scheduler::add_task(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_){

	//add the task to the legacy schedule object, but also add to vector
	//to make the scheduler much easier to read and work with.
	auto taskData_object = schedule.add_task(elasticity_, num_modes_, work_, span_, period_, gpu_work_, gpu_span_, gpu_period_);

	task_table.push_back(std::vector<task_mode>());

	std::cout << "Task Losses:" << std::endl; 

	for (int j = 0; j < num_modes_; j++){

		task_mode item;

		//the loss function is different if the 
		//task is a pure cpu task or hybrid task
		if (taskData_object->pure_cpu_task())
			item.cpuLoss = (1.0 / taskData_object->get_elasticity() * (std::pow(taskData_object->get_max_utilization() - (taskData_object->get_work(j) / taskData_object->get_period(j)), 2)));
		
		else 
			item.cpuLoss = (1.0 / taskData_object->get_elasticity() * (std::pow(taskData_object->get_max_utilization() - ((taskData_object->get_work(j) / taskData_object->get_period(j)) + (taskData_object->get_GPU_work(j) / taskData_object->get_period(j))), 2)));


		std::cout << "Mode "<< j << " Loss: " << item.cpuLoss << std::endl;

		item.gpuLoss = 0;
		item.cores = taskData_object->get_CPUs(j);
		item.sms = taskData_object->get_GPUs(j);

		task_table.at(task_table.size() - 1).push_back(item);

		//check all other modes stored for this task
		//and if it gains one resource while losing another
		//mark it as unsafe
		for (int i = 0; i < task_table.at(task_table.size() - 1).size(); i++){

			for (int k = i; k < task_table.at(task_table.size() - 1).size(); k++){

				if (task_table.at(task_table.size() - 1).at(i).cores < task_table.at(task_table.size() - 1).at(k).cores && task_table.at(task_table.size() - 1).at(i).sms > task_table.at(task_table.size() - 1).at(k).sms)
					task_table.at(task_table.size() - 1).at(i).unsafe_mode = true;

				else if (task_table.at(task_table.size() - 1).at(i).cores > task_table.at(task_table.size() - 1).at(k).cores && task_table.at(task_table.size() - 1).at(i).sms < task_table.at(task_table.size() - 1).at(k).sms)
					task_table.at(task_table.size() - 1).at(i).unsafe_mode = true;

			}

		}
		
	}

	return taskData_object;
}

//static vector sizes
std::vector<std::vector<Scheduler::task_mode>> Scheduler::task_table(100, std::vector<task_mode>(100));

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
    
    //Legacy code, but nice to see the counts
    int provider_order = 0;
    int consumer_order = 0;
	int transfer_order = 0;

	std::ostringstream mode_strings;

	print_module::buffered_print(mode_strings, "\n========================= \n", "Nodes Passed to RAG Builder:\n");
    
    for (const auto& [id, node] : nodes) {

		print_module::buffered_print(mode_strings, "Node ", id, " <", node.x, ",", node.y, ">\n");

		if (node.x == 0 && node.y == 0)
			continue;
    
	    if (node.x >= 0 && node.y >= 0)
            provider_order += 1;
    
	    if (node.x <= 0 && node.y <= 0)
            consumer_order += 1;

		if ((node.x < 0 && node.y > 0) || (node.x > 0 && node.y < 0))
			transfer_order += 1;

    }

	print_module::buffered_print(mode_strings, "=========================\n\n");
	print_module::flush(std::cerr, mode_strings);

	print_module::print(std::cerr, "Provider size: ", provider_order, " Consumer size: ", consumer_order, " Transfer size: ", transfer_order, "\n");
    
	//if system has barrier, just do it lazily
	if (barrier){

		for (int consumer_id = 0; consumer_id < (int) nodes.size(); consumer_id++){

			Node& consumer = nodes[consumer_id];
			int needed_x = -consumer.x;
			int needed_y = -consumer.y;
			
			for (int provider_id = 0; provider_id < (int) nodes.size(); provider_id++) {

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

					//Update provider's available resources
					provider.x -= new_edge.x_amount;
					provider.y -= new_edge.y_amount;
				

				}

			}

		}

		return true;
	}

    //Just build the RAG the same way we did when 
	//we discovered the solution
	std::vector<int> discovered_providers;
	std::vector<int> discovered_consumers;
	std::vector<int> discovered_transformers;

	//reserve
	discovered_providers.reserve(nodes.size());
	discovered_consumers.reserve(nodes.size());
	discovered_transformers.reserve(nodes.size());

	//add the free pool to the providers
	discovered_providers.push_back(nodes.size() - 1);

	//loop and discover all nodes and fix transformers
	for (int i = 0; i < (int) nodes.size() - 1; i++){

		Node& node = nodes[i];

		if (node.x == 0 && node.y == 0)
			continue;

		//if it's a consumer, just add it to the discovered
		if (node.x <= 0 && node.y <= 0){

			discovered_consumers.push_back(i);

			std::cout << "Consumer: " << i << std::endl;

		}

		//if a pure provider, add it to the list as well
		else if (node.x >= 0 && node.y >= 0){

			discovered_providers.push_back(i);

			std::cout << "Provider: " << i << std::endl;

		}

		//otherwise, it's a transformer
		else {
			
			discovered_transformers.push_back(i);

			std::cout << "Transformer: " << i << std::endl;
		}

	}

	//now check if there is a way to solve this
	//chain of trades
	std::pair<int, int> resource_pool;

	//add all providers to the resource pool
	for (int provider_id : discovered_providers){

		Node& provider = nodes[provider_id];
		resource_pool.first += provider.x;
		resource_pool.second += provider.y;

	}

	//split the transformer pool in 2 based on resources
	std::vector<int> transformer_pool_x;
	std::vector<int> transformer_pool_y;

	for (int transformer_id : discovered_transformers){

		Node& transformer = nodes[transformer_id];

		if (transformer.x > 0)
			transformer_pool_x.push_back(transformer_id);

		if (transformer.y > 0)
			transformer_pool_y.push_back(transformer_id);

	}

	//now start the small knapsack problems
	int failed = 0;
	int cycle = 0;

	//vector to hold which elements we want to process
	//separated by -1 to indicate chain swap
	std::vector<int> processing_chain;

	int elements_removed_x = 0;
	int elements_removed_y = 0;

	while (failed != 2 && (transformer_pool_x.size() != elements_removed_x && transformer_pool_y.size() != elements_removed_y)){

		int transformers_ct = (cycle % 2 == 0) ? transformer_pool_x.size() : transformer_pool_y.size();
		int processor_ct = (cycle % 2 == 0) ? resource_pool.first : resource_pool.second;

		double dp_two[transformers_ct + 1][processor_ct + 1][3];
		int solutions[transformers_ct + 1][processor_ct + 1][transformers_ct];

		//memset them all to 0
		memset(dp_two, 0, sizeof(double) * (transformers_ct + 1) * (processor_ct + 1) * 3);
		memset(solutions, -1, sizeof(int) * (transformers_ct + 1) * (processor_ct + 1) * transformers_ct);

		int skipped_nodes = 0;

		//Execute exact solution
		for (int i = 1; i <= transformers_ct; i++) {

			auto node_id = (cycle % 2 == 0) ? transformer_pool_x.at(i - 1) : transformer_pool_y.at(i - 1);

			if (node_id == -1){

				skipped_nodes += 1;
				continue;

			}

			for (int w = 0; w <= processor_ct; w++) {

				int item_value;
				int item_weight;

				//fetch correct item
				if (cycle % 2 == 0){
					
					item_value = nodes[node_id].x;
					item_weight = nodes[node_id].y * -1;

				}

				else {

					item_value = nodes[node_id].y;
					item_weight = nodes[node_id].x * -1;
					
				}

				//if this item is feasible at all 
				if ((w >= item_weight)){

					double newValue = dp_two[i - 1][w - item_weight][0] + item_value;
					
					//if found solution is better, update
					if ((newValue) > (dp_two[i][w][0])) {

						dp_two[i][w][0] = newValue;

						std::memcpy(solutions[i][w], solutions[i - 1][w - item_weight], num_tasks * sizeof(int));

						solutions[i][w][i - 1] = i;

					}

					else{
						
						dp_two[i][w][0] = dp_two[i - 1][w][0];

						std::memcpy(solutions[i][w], solutions[i - 1][w], num_tasks * sizeof(int));

					}
				}
			}
		}

		//check if we have a valid solution
		std::vector<int> result;
		double gained_resources = 0;

		for (int i = 0; i < (int) num_tasks; i++)
			if (solutions[transformers_ct][processor_ct][i] != -1)
				result.push_back(solutions[transformers_ct][processor_ct][i]);

		gained_resources = dp_two[transformers_ct][processor_ct][0];

		//check to see that we got a solution to progress the problem
		if ((result.size() == 0 || gained_resources == 0) && (transformers_ct != skipped_nodes)){

			failed += 1;

		}

		else {

			//update that the solution is progressing
			if (failed > 0)
				failed -= 1;

			//add the discovered chain to the processing chain
			if (cycle % 2 == 0)
				for (int i = 0; i < (int) result.size(); i++)
					processing_chain.push_back(transformer_pool_x.at(result.at(i)));

			else
				for (int i = 0; i < (int) result.size(); i++)
					processing_chain.push_back(transformer_pool_y.at(result.at(i)));

			processing_chain.push_back(-1);

			//now remove these nodes from the transformer pool by setting them to -1
			for (int i = 0; i < (int) result.size(); i++){

				if (cycle % 2 == 0){

					transformer_pool_x.at(result.at(i)) = -1;
					elements_removed_x += 1;

				}

				else {

					transformer_pool_y.at(result.at(i)) = -1;
					elements_removed_y += 1;

				}

			}

		}

	}

	//at the end of this, the processing chain should be filled with the order
	//of nodes to process. Then we just process the consumers like we did before.
	//If we fail, we will have to do multiple mode changes to satisfy the requirements
	if (failed == 2){

		print_module::print(std::cerr, "Error: System is not schedulable in any configuration. Exiting.\n");
		killpg(process_group, SIGINT);
		return false;

	}

	//loop and fix transformers in the chain
	for (int i = 0; i < (int) processing_chain.size(); i++){

		Node& consumer = nodes[processing_chain.at(i)];
		int needed_x = -consumer.x;
		int needed_y = -consumer.y;

		for (int provider_id : discovered_providers) {
		
			Node& provider = nodes[provider_id];
			Edge new_edge{i, 0, 0};
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

				//Update provider's available resources
				provider.x -= new_edge.x_amount;
				provider.y -= new_edge.y_amount;

			}

		}

		//now this once transformer is a provider
		discovered_providers.push_back(processing_chain.at(i));
	}

	//now just do the same thing we did for transformers
	//but with the discovered consumers
	for (int consumer_id : discovered_consumers){

		Node& consumer = nodes[consumer_id];
		int needed_x = -consumer.x;
		int needed_y = -consumer.y;

		for (int provider_id : discovered_providers) {
		
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

				//Update provider's available resources
				provider.x -= new_edge.x_amount;
				provider.y -= new_edge.y_amount;

			}

		}
		
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

	#ifdef __NVCC__

		if (first_time)
			CUDA_SAFE_CALL(cuInit(0));

	#endif

	//vector for transitioned tasks
	std::vector<int> transitioned_tasks;

	//for each run we need to see what resources are left in the pool from the start
	int starting_CPUs = free_cores_A.size();
	int starting_GPUs = free_cores_B.size();

	if (first_time) {

		//add an entry for each task into previous modes
		for (int i = 0; i < schedule.count(); i++)
			previous_modes.push_back(task_mode());

		//copy all of the task table to the device after conversion

		#ifdef __NVCC__

			//25 tasks, 4 modes (0-3): cores, sms, cpuLoss
			int host_task_table[25 * 8 * 2];
			double host_losses[25 * 8];

			//find largest number of modes in the task table
			int max_modes = 0;

			for (int i = 0; i < (int) task_table.size(); i++)
				if ((int) task_table.at(i).size() > max_modes)
					max_modes = (int) task_table.at(i).size();

			for (int i = 0; i < (int) task_table.size(); i++){

				for (int j = 0; j < (int) task_table.at(i).size(); j++){

					host_task_table[i * 8 * 2 + j * 2 + 0] = task_table.at(i).at(j).cores;
					host_task_table[i * 8 * 2 + j * 2 + 1] = task_table.at(i).at(j).sms;
					host_losses[i * 8 + j] = task_table.at(i).at(j).cpuLoss;

				}

				//if this task had fewer modes than max, pad all the rest with -1
				for (int j = (int) task_table.at(i).size(); j < max_modes; j++){

					host_task_table[i * 8 * 2 + j * 2 + 0] = -1;
					host_task_table[i * 8 * 2 + j * 2 + 1] = -1;
					host_losses[i * 8 + j] = -1;

				}

			}

			//get the symbol on the device
			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&d_task_table, sizeof(int) * 25 * 8 * 2));
			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&d_uncooperative_tasks, sizeof(int) * 25));
			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&d_final_solution, sizeof(int) * 25));
			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&d_losses, sizeof(double) * 25 * 8));
			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&d_final_loss, sizeof(double)));

			//copy it
			CUDA_NEW_SAFE_CALL(cudaMemcpy(d_task_table, host_task_table, sizeof(int) * 25 * 8 * 2, cudaMemcpyHostToDevice));
			CUDA_NEW_SAFE_CALL(cudaMemcpy(d_losses, host_losses, sizeof(double) * 25 * 8, cudaMemcpyHostToDevice));

			CUDA_NEW_SAFE_CALL(cudaMemcpyToSymbol(constant_task_table, &host_task_table, sizeof(int) * 25 * 8 * 2));
			CUDA_NEW_SAFE_CALL(cudaMemcpyToSymbol(constant_losses, &host_losses, sizeof(double) * 25 * 8));

			set_dp_table<<<1, 1>>>();

			CUDA_NEW_SAFE_CALL(cudaDeviceSynchronize());

		#endif

	}

	//dynamic programming table
	int N = task_table.size();
	std::vector<int> best_solution;

	//force checks to ensure each task has their core count
	if (!first_time){

		int total_cores = 0;
		int total_gpus = 0;

		for (int i = 0; i < (int) previous_modes.size(); i++){
			
			//fetch this task's current cores
			//(-1 because perm core is never returned)
			auto task_owned_cpus = (schedule.get_task(i))->get_cpu_owned_by_process();

			if (((previous_modes.at(i).cores - 1) != (int) task_owned_cpus.size())){
				
				std::cout << "CPU Count Mismatch. Process:" << i << " | Cores assigned: " << previous_modes.at(i).cores << " | Cores found: " << task_owned_cpus.size() << " | Cannot Continue" << std::endl;
				killpg(process_group, SIGINT);
				return;

			}
			
			//get sm units
			//no -1 because no perm gpu
			auto task_owned_gpus = (schedule.get_task(i))->get_gpu_owned_by_process();

			if ((previous_modes.at(i).sms) != (int) task_owned_gpus.size()){

				std::cout << "GPU Count Mismatch. Process:" << i << " | GPUs assigned: " << previous_modes.at(i).sms << " | GPUs found: " << task_owned_gpus.size() << " | Cannot Continue" << std::endl;
				killpg(process_group, SIGINT);
				return;

			}

			//add to total
			total_cores += previous_modes.at(i).cores;
			total_gpus += previous_modes.at(i).sms;

		}

		//check that the total cores in the system - the total found is the free count
		if (((int) maxCPU - total_cores) != (int) free_cores_A.size()){

			std::cout << "CPU Count Mismatch. Total Cores: " << maxCPU << " | Total Found: " << total_cores << " | Free Cores: " << free_cores_A.size() << " | Cannot Continue" << std::endl;
			killpg(process_group, SIGINT);
			return;

		}

		//check that the total gpus in the system - the total found is the free count
		if (((int) NUMGPUS - total_gpus) != (int) free_cores_B.size()){

			std::cout << "GPU Count Mismatch. Total GPUs: " << NUMGPUS << " | Total Found: " << total_gpus << " | Free GPUs: " << free_cores_B.size() << " | Cannot Continue" << std::endl;
			killpg(process_group, SIGINT);
			return;

		}

	}

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
			if (((int)(NUMGPUS) - min_required_gpu + (schedule.get_task(i))->get_min_GPUs()) < (schedule.get_task(i))->get_max_GPUs())
				(schedule.get_task(i))->set_practical_max_GPUs( NUMGPUS - min_required_gpu + (schedule.get_task(i))->get_min_GPUs());

			else
				(schedule.get_task(i))->set_practical_max_GPUs((schedule.get_task(i))->get_max_GPUs());

		}
	}

	//get current time
	timespec start_time;
	clock_gettime(CLOCK_MONOTONIC, &start_time);

	double loss;

	#ifdef __NVCC__

		//copy over all the uncooperative tasks' selected modes
		int host_uncooperative[25] = {0};
		for (int i = 0; i < schedule.count(); i++)
			if (!(schedule.get_task(i))->get_changeable() || !(schedule.get_task(i))->cooperative())
				host_uncooperative[i] = (schedule.get_task(i))->get_current_mode();

		//copy the array to the device
		CUDA_NEW_SAFE_CALL(cudaMemcpy(d_uncooperative_tasks, host_uncooperative, 25 * sizeof(int), cudaMemcpyHostToDevice));

		//Execute exact solution
		device_do_schedule<<<1, 1024>>>(N, maxCPU, NUMGPUS, d_task_table, d_losses, d_final_loss, d_uncooperative_tasks, d_final_solution);

		//peek for launch errors
		CUDA_NEW_SAFE_CALL(cudaPeekAtLastError());

		//sync
		CUDA_NEW_SAFE_CALL(cudaDeviceSynchronize());

		//copy the final_solution array back
		int host_final[25] = {0};

		//copy it 
		CUDA_NEW_SAFE_CALL(cudaMemcpy(host_final, d_final_solution, 25 * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_NEW_SAFE_CALL(cudaMemcpy(&loss, d_final_loss, sizeof(double), cudaMemcpyDeviceToHost));

	#else

		double dp_two[N + 1][maxCPU + 1][NUMGPUS + 1][3];
		int solutions[num_tasks + 1][maxCPU + 1][NUMGPUS + 1][num_tasks];

		for(int i = 0; i <= (int) num_tasks; i++) {

			for(int j = 0; j <= (int) maxCPU; j++) {

				for(int k = 0; k <= (int) NUMGPUS; k++) {

					dp_two[i][j][k][0] = 100000;
					dp_two[i][j][k][1] = starting_CPUs;
					dp_two[i][j][k][2] = starting_GPUs;

				}

			}

		}

		//Execute exact solution
		for (int i = 1; i <= (int) num_tasks; i++) {

			for (int w = 0; w <= (int) maxCPU; w++) {

				for (int v = 0; v <= (int) NUMGPUS; v++) {

					//invalid state
					dp_two[i][w][v][0] = -1.0;
					dp_two[i][w][v][1] = 0.0;
					dp_two[i][w][v][2] = 0.0;
					
					int j_start = 0;
					int j_end = (int) task_table.at(i - 1).size();

					//check if cooperative
					if (!(schedule.get_task(i - 1))->get_changeable() || !(schedule.get_task(i - 1))->cooperative()){

						j_start = (schedule.get_task(i - 1))->get_current_mode();
						j_end = j_start + 1;

					}

					//for each item in class
					for (size_t j = j_start; j < j_end; j++) {

						auto item = task_table.at(i - 1).at(j);

						//fetch initial suspected resource values
						int current_item_sms = item.sms;
						int current_item_cores = item.cores;

						//if this item is feasible at all 
						if ((w >= current_item_cores) && (v >= current_item_sms) && (dp_two[i - 1][w - current_item_cores][v - current_item_sms][0] != -1)){

							//if item fits in both sacks
							if ((w >= current_item_cores) && (v >= current_item_sms) && (dp_two[i - 1][w - current_item_cores][v - current_item_sms][0] != -1)) {

								double newCPULoss_two = dp_two[i - 1][w - current_item_cores][v - current_item_sms][0] - item.cpuLoss;
								
								//if found solution is better, update
								if ((newCPULoss_two) > (dp_two[i][w][v][0])) {

									dp_two[i][w][v][0] = newCPULoss_two;

									std::memcpy(solutions[i][w][v], solutions[i - 1][w - current_item_cores][v - current_item_sms], num_tasks * sizeof(int));

									solutions[i][w][v][i - 1] = j;

								}
							}
						}
					}
				}
			}
		}

	#endif

    //return optimal solution
	std::vector<int> result;

	for (int i = 0; i < (int) num_tasks; i++) {
		

		#ifdef __NVCC__

			if (host_final[i] != -1)
				result.push_back(host_final[i]);

		#else

			if (solutions[num_tasks][(int) maxCPU][(int) NUMGPUS][i] != -1)
				result.push_back(solutions[num_tasks][(int) maxCPU][(int) NUMGPUS][i]);
		
		#endif

	}

	#ifndef __NVCC__

		loss = 100000 - dp_two[num_tasks][(int) maxCPU][(int) NUMGPUS][0];

	#endif

	//check to see that we got a solution that renders this system schedulable
	if ((result.size() == 0 || loss == 100001) && first_time){

		print_module::print(std::cerr, "Error: System is not schedulable in any configuration. Exiting.\n");
		killpg(process_group, SIGINT);
		return;

	}
	
	else if ((result.size() == 0 || loss == 100001)){


		print_module::print(std::cerr, "Error: System is not schedulable in any configuration with specified constraints. Not updating modes.\n");
		return;

	}

	//deal with pessemism negatives
	auto backup = result;

	timespec end_time;
	clock_gettime(CLOCK_MONOTONIC, &end_time);

	//determine ellapsed time in nanoseconds
	double elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1e9;
	elapsed_time += (end_time.tv_nsec - start_time.tv_nsec);

	//print out the time taken
	print_module::print(std::cerr, "Time taken to run just the double knapsack: ", elapsed_time / 1000000, " milliseconds.\n");

	//update the tasks
	std::ostringstream mode_strings;
	print_module::buffered_print(mode_strings, "\n========================= \n", "New Schedule Layout:\n");
	for (size_t i = 0; i < result.size(); i++)
		print_module::buffered_print(mode_strings, "Task ", i, " is now in mode: ", result.at(i), "\n");
	print_module::buffered_print(mode_strings, "Total Loss from Mode Change: ", loss, "\n=========================\n\n");

	//print resources now held by each task
	print_module::buffered_print(mode_strings, "\n========================= \n", "New Resource Layout:\n");
	for (size_t i = 0; i < result.size(); i++)
		print_module::buffered_print(mode_strings, "Task ", i, " now has: ", task_table.at(i).at(result.at(i)).cores, " Core A | ", task_table.at(i).at(result.at(i)).sms, " Core B\n");
	print_module::buffered_print(mode_strings, "=========================\n\n");
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
				killpg(process_group, SIGINT);
                return;

			}

			(schedule.get_task(i))->set_current_lowest_CPU(next_CPU);
			next_CPU += (schedule.get_task(i))->get_current_CPUs();

			if (next_CPU > num_CPUs + 1){

				print_module::print(std::cerr, "Error in task ", i, ": too many CPUs have been allocated.", next_CPU, " ", num_CPUs, " \n");
				killpg(process_group, SIGINT);
				return;

			}		

		}

		//assign all the unassigned cpus to the scheduler to hold
		for (int i = next_CPU; i < num_CPUs; i++){

			print_module::print(std::cerr, "CPU ", i, " is free.\n");
			free_cores_A.push_back(i);

		}
		//Now assign TPC units to tasks, same method as before
		//(don't worry about holding TPC 1) 
		int next_TPC = 0;

		for (int i = 0; i < schedule.count(); i++){

			if ((schedule.get_task(i))->get_current_lowest_GPU() > 0){

				print_module::print(std::cerr, "Error in task ", i, ": all tasks should have had lowest GPU cleared. (this likely means memory was not cleaned up)\n");
				killpg(process_group, SIGINT);
				return;

			}

			//if this task actually has any TPCs assigned
			if (!(schedule.get_task(i))->pure_cpu_task()){

				(schedule.get_task(i))->set_current_lowest_GPU(next_TPC);

				for (int j = 0; j < (schedule.get_task(i))->get_current_GPUs(); j++)
					(schedule.get_task(i))->push_back_gpu(next_TPC ++);

				if (next_TPC > (int)(NUMGPUS) + 1){

					print_module::print(std::cerr, "Error in task ", i, ": too many GPUs have been allocated.", next_TPC, " ", NUMGPUS, " \n");
					killpg(process_group, SIGINT);
					return;

				}

			}
		}

		//assign all the unassigned gpus to the scheduler to hold
		for (int i = next_TPC; i < (int)(NUMGPUS); i++)
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

		result = backup;

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
					if ((previous_mode.cores - current_mode.cores) > 0){

						if (id != ((int) nodes.size() - 1) && (int) task_owned_cpus.size() < (previous_mode.cores - current_mode.cores)){

							print_module::print(std::cerr, "Error: not enough CPUs to give to task free pool from task ", id, " size gotten: ", task_owned_cpus.size(), " expected: ", (previous_mode.cores - current_mode.cores), ". Exiting.\n");
							killpg(process_group, SIGINT);
							return;

						}

						for (int i = 0; i < (previous_mode.cores - current_mode.cores); i++){

							free_cores_A.push_back(task_owned_cpus.at(task_owned_cpus.size() - 1));
							task_owned_cpus.pop_back();
						
						}

						auto change_amount = (schedule.get_task(id))->get_CPUs_change();
						(schedule.get_task(id))->set_CPUs_change(change_amount + (previous_mode.cores - current_mode.cores));
						

					}

					if ((previous_mode.sms - current_mode.sms) > 0){

						for (int i = 0; i < (previous_mode.sms - current_mode.sms); i++){

							free_cores_B.push_back(task_owned_gpus.at(task_owned_gpus.size() - 1));
							task_owned_gpus.pop_back();
						
						}

						auto change_amount = (schedule.get_task(id))->get_GPUs_change();
						(schedule.get_task(id))->set_GPUs_change(change_amount + (previous_mode.sms - current_mode.sms));
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

							//update our own
							CPUs_given_up += edge.x_amount;

							if(id == 10){
								
								print_module::print(std::cerr, "Task ", id, " is giving ", edge.x_amount, " CPUs to task ", task_being_given_to, ".\n");
							}

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

							//update our own
							GPUs_given_up += edge.y_amount;

						}

					}

				}

				//check if we gave up resource AND we are giving resources to the free pool
				if (id != ((int) nodes.size() - 1)){ 

					if ((previous_mode.cores - current_mode.cores) > CPUs_given_up){

					
						if (((previous_mode.cores - current_mode.cores) - CPUs_given_up) > (int) task_owned_cpus.size()){

							print_module::print(std::cerr, "Error: not enough CPUs to give to free pool from task ", id, ". size gotten: ", task_owned_cpus.size(), " expected: ", ((previous_mode.cores - current_mode.cores) - CPUs_given_up), ". Exiting.\n");
							killpg(process_group, SIGINT);
							return;

						}

						for (int i = CPUs_given_up; i < (previous_mode.cores - current_mode.cores); i++){

							free_cores_A.push_back(task_owned_cpus.at(task_owned_cpus.size() - 1));
							task_owned_cpus.pop_back();

							CPUs_given_up++;
							
						}

					}

					if ((previous_mode.sms - current_mode.sms) > GPUs_given_up){

					
						if (((previous_mode.sms - current_mode.sms) - GPUs_given_up) > (int) task_owned_gpus.size()){

							print_module::print(std::cerr, "Error: not enough GPUs to give to free pool from task ", id, ". size gotten: ", task_owned_gpus.size(), " expected: ", ((previous_mode.sms - current_mode.sms) - GPUs_given_up), ". Exiting.\n");
							killpg(process_group, SIGINT);
							return;

						}

						for (int i = GPUs_given_up; i < (previous_mode.sms - current_mode.sms); i++){

							free_cores_B.push_back(task_owned_gpus.at(task_owned_gpus.size() - 1));

							task_owned_gpus.pop_back();

							GPUs_given_up++;
							
						}

					}

				}
				
				//let the task know what it should give up when it can change modes
				(schedule.get_task(id))->set_CPUs_change(CPUs_given_up);
				(schedule.get_task(id))->set_GPUs_change(GPUs_given_up);

				//add the task to list of tasks that had to transition
				transitioned_tasks.push_back(id);

			}
		}	

		else{

			if (!barrier){

				print_module::print(std::cerr, "Error: System was passed a RAG to build a DAG with, but a cycle was detected, and no barrier is enabled. Not updating states.\n");
				return;

			}
		
		}
 
	}

	//update the previous modes to the current modes
	for (size_t i = 0; i < result.size(); i++){

		previous_modes.at(i) = task_table.at(i).at(result.at(i));

		//notify all tasks that they should now transition
		(schedule.get_task(i))->set_mode_transition(false);

	}

	first_time = false;

	clock_gettime(CLOCK_MONOTONIC, &end_time);

	//determine ellapsed time in nanoseconds
	elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1e9;
	elapsed_time += (end_time.tv_nsec - start_time.tv_nsec);

	//print out the time taken
	print_module::print(std::cerr, "Time taken to reschedule: ", elapsed_time / 1000000, " milliseconds.\n");

}

void Scheduler::setTermination(){
	schedule.setTermination();
}