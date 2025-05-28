#include <iostream>
#include <string>

#ifdef __NVCC__

	#include <cuda.h>
	#include <cuda_runtime.h>

	#define CUDA_SAFE_CALL(x)                                        	\
	do {                                                              	\
	CUresult result = x;                                           		\
	if (result != CUDA_SUCCESS) {                                  		\
	const char *msg;                                            		\
		cuGetErrorName(result, &msg);                               	\
		std::cerr << "\nerror: " #x " failed with error "           	\
					<< msg << '\n';                                   	\
		exit(1);                                                    	\
	}                                                              		\
	} while(0)

	#define CUDA_NEW_SAFE_CALL(x)                                    	\
	do {                                                              	\
	cudaError_t result = x;                                        		\
	if (result != cudaSuccess) {                                   		\
	std::cerr << "\nerror: " #x " failed with error "           		\
					<< cudaGetErrorName(result) << '\n';              	\
		exit(1);                                                    	\
	}                                                              		\
	} while(0)


	#define HOST_DEVICE_BLOCK_DIM blockDim.x 
	#define HOST_DEVICE_THREAD_DIM threadIdx.x
	#define HOST_DEVICE_SCOPE __device__
	#define HOST_DEVICE_CONSTANT __constant__
	#define HOST_DEVICE_GLOBAL __global__
	#define HOST_DEVICE_SHARED __shared__

#else 

	#define HOST_DEVICE_BLOCK_DIM 1
	#define HOST_DEVICE_THREAD_DIM 0
	#define HOST_DEVICE_SCOPE
	#define HOST_DEVICE_CONSTANT
	#define HOST_DEVICE_GLOBAL static
	#define HOST_DEVICE_SHARED static

#endif

//HOST_DEVICE_SHARED float shared_dp_two[2][64 + 1][64 + 1];

HOST_DEVICE_SCOPE volatile int solutions[MAXTASKS][128 + 1][128 + 1];

//HOST_DEVICE_SCOPE volatile char free_resource_pool[2][64 + 1][64 + 1][2];

HOST_DEVICE_CONSTANT int constant_task_table[MAXTASKS * MAXMODES * 3];

HOST_DEVICE_CONSTANT float constant_losses[MAXTASKS * MAXMODES];

HOST_DEVICE_SCOPE void zero_one_knapsack(int* item_weights, int* item_values, int* out_array, int num_items, int max_weight) {

	//FIXME: WHEN THERE IS ONE ITEM PASSED, WE END UP GETTING
	//THE WRONG VALUE OUT IF THE ITEM CANNOT BE TAKEN

	//FIXME: MIGHT BE BETTER TO NOT EVER SET THE TABLE AND USE 
	//STATIC VARIABLES WHEN i == 1
	int dp_table[10 + 1][128 + 1];
	for (int i = 0; i <= max_weight; i++)
		dp_table[0][i] = 0;

	for (int i = 1; i <= num_items; i++) {

		for (int w = 0; w <= max_weight; w++) {
		    
		    if (item_weights[i - 1] == -1){
    		    dp_table[i][w] = dp_table[(i - 1)][w];
    		    continue;
    		}

			dp_table[i][w] = dp_table[i - 1][w];

			if (item_weights[i - 1] <= w){

				int old_value = dp_table[(i - 1)][w];
				int new_value = dp_table[(i - 1)][w - item_weights[i - 1]] + item_values[i - 1];

				dp_table[i][w] = (new_value > old_value) ? new_value : old_value;

			}
			
			else
				dp_table[i][w] = dp_table[(i - 1)][w];

		}

	}

	//backtrack to find the items
	int w = max_weight;
	for (int i = num_items; i > 0; i--) {

		if (dp_table[i][w] != dp_table[i - 1][w]) {

			out_array[i - 1] = 1;
			w -= item_weights[i - 1];

		} 
		
		else {

			out_array[i - 1] = 0;

		}

	}

}

HOST_DEVICE_SCOPE bool rebuild_solution(int* final_solution, int* loopback_indices, int num_tasks, int maxCPU, int NUMGPUS) {

	bool valid_solution = true;
	int current_w = maxCPU;
	int current_v = NUMGPUS;

	for (int i = num_tasks; i > 0; i--){

		int group_idx = loopback_indices[i - 1];

		int current_item = solutions[i][current_w][current_v];

		if (current_item == -1){

			valid_solution = false;

		}

		else {

			//take the core and sm values for the item
			int current_item_cores = constant_task_table[(group_idx - 1) * MAXMODES * 3 + current_item * 3];
			int current_item_sms = constant_task_table[(group_idx - 1) * MAXMODES * 3 + current_item * 3 + 1];

			//update the current w and v values
			current_w = current_w - current_item_cores;
			current_v = current_v - current_item_sms;
			
		}

		final_solution[i - 1] = current_item;

	}

	return valid_solution;

}

HOST_DEVICE_SCOPE bool reorder_heuristic(int* current_solution, int* task_table, int num_items, int* slack_A, int* slack_B, int check_x_first){

	//The solution currently being reordered is passed to us as an int* array where -1 means the item
	//is not currently being used in the solution. This is to enable the reorder buffer, so we
	//can easily extend each of these modules to item skipping in the next phase

	//now we need to classify the items in the current solution as either x providers or 
	//y providers, or both
	int pure_providers[MAXTASKS];

	int x_providers_weight[MAXTASKS];
	int y_providers_weight[MAXTASKS];

	int x_providers_value[MAXTASKS];
	int y_providers_value[MAXTASKS];

	int x_providers_count = 0;
	int y_providers_count = 0;
	int pure_providers_count = 0;

	for (int i = 0; i < num_items; i++) {

		int item = current_solution[i];

		if (item == -1)
			continue;

		int item_current_cores = task_table[i * 2];
		int item_current_sms = task_table[(i * 2) + 1];

		int solution_item_sms = constant_task_table[(i) * MAXMODES * 3 + item * 3 + 1];
		int solution_item_cores = constant_task_table[(i) * MAXMODES * 3 + item * 3];

		int delta_cores = item_current_cores - solution_item_cores;
		int delta_sms = item_current_sms - solution_item_sms;

		if (delta_cores > 0 && delta_sms > 0){

			pure_providers[pure_providers_count++] = i;

		} 
		
		else if (delta_cores > 0 && delta_sms < 0){

			x_providers_weight[x_providers_count] = -delta_sms;

			x_providers_value[x_providers_count++] = delta_cores;

		} 
		
		else if (delta_cores < 0 && delta_sms > 0){

			y_providers_weight[y_providers_count] = -delta_cores;

			y_providers_value[y_providers_count++] = delta_sms;

		}

	}

	//after this all items are classified and consumers are ignored
	//now we need to reorder the items in the solution. We have to try
	//starting from both sides of the problem to ensure there is no solution.
	//which one is better is unfortunately undecideable, so we will just go
	//with the first one that works
	int current_slack_A = *slack_A;
	int current_slack_B = *slack_B;

	//get the slack out of the providers
	for (int i = 0; i < pure_providers_count; i++) {

		int item = pure_providers[i];

		int item_current_cores = task_table[item * 2];
		int item_current_sms = task_table[(item * 2) + 1];

		int solution_item_sms = constant_task_table[(item) * MAXMODES * 3 + item * 3 + 1];
		int solution_item_cores = constant_task_table[(item) * MAXMODES * 3 + item * 3];

		current_slack_A += (item_current_cores - solution_item_cores);
		current_slack_B += (item_current_sms - solution_item_sms);

	}

	//we now start with whatever provider we are positioning in this
	//iteration

	int x_providers_processed = 0;
	int y_providers_processed = 0;

	int out_array[MAXTASKS];

	bool found_x = true;
	bool found_y = true;

	//handle which one we should start with
	/*if (check_x_first){

		zero_one_knapsack(x_providers_weight, x_providers_value, out_array, x_providers_count, current_slack_B);

		//we take the items the knapsack solver found,
		//and we remove the slack they used, and add the slack they
		//provided. We then update the processed numbers and remove 
		//them from the x_providers arrays
		for (int i = 0; i < x_providers_count; i++) {

			if (out_array[i] == 1){

				int value = x_providers_value[i];
				int weight = x_providers_weight[i];

				current_slack_A += value;
				current_slack_B -= weight;

				x_providers_processed++;

				//remove the item from the x_providers arrays
				x_providers_value[i] = -1;
				x_providers_weight[i] = -1;

			}

		}

	}

	else {

		zero_one_knapsack(y_providers_weight, y_providers_value, out_array, y_providers_count, current_slack_A);

		//we take the items the knapsack solver found,
		//and we remove the slack they used, and add the slack they
		//provided. We then update the processed numbers and remove
		//them from the y_providers arrays
		for (int i = 0; i < y_providers_count; i++) {

			if (out_array[i] == 1){

				int value = y_providers_value[i];
				int weight = y_providers_weight[i];

				current_slack_A -= value;
				current_slack_B += weight;

				y_providers_processed++;

				//remove the item from the y_providers arrays
				y_providers_value[i] = -1;
				y_providers_weight[i] = -1;


			}

		}

	}*/

	//now just repeat until we either get a loop
	//or we process all elements
	while(x_providers_processed != x_providers_count && y_providers_processed != y_providers_count && (found_x || found_y)) {

		found_x = false;
		found_y = false;

		//start the knapsack
		zero_one_knapsack(x_providers_weight, x_providers_value, out_array, x_providers_count, current_slack_B);

		//we take the items the knapsack solver found,
		//and we remove the slack they used, and add the slack they
		//provided. We then update the processed numbers and remove 
		//them from the x_providers arrays
		for (int i = 0; i < x_providers_count; i++) {

			if (out_array[i] == 1){

				int value = x_providers_value[i];
				int weight = x_providers_weight[i];

				current_slack_A += value;
				current_slack_B -= weight;

				if (current_slack_B < 0){

					//if we are here, we have a problem
					std::cout << "We have a problem with the reorder heuristic, current slack B is negative." << std::endl;
					std::cout << "Current slack A: " << current_slack_A << std::endl;
					std::cout << "Current slack B: " << current_slack_B << std::endl;

					//print out array
					std::cout << "Out array: " << std::endl;
					for (int j = 0; j < x_providers_count; j++) {
						std::cout << out_array[j] << " ";
					}

				}

				x_providers_processed++;

				//remove the item from the x_providers arrays
				x_providers_value[i] = -1;
				x_providers_weight[i] = -1;

				//update the bool
				found_x = true;

			}

		}

		//run the knapsack for the y providers
		zero_one_knapsack(y_providers_weight, y_providers_value, out_array, y_providers_count, current_slack_A);

		//we take the items the knapsack solver found,
		//and we remove the slack they used, and add the slack they
		//provided. We then update the processed numbers and remove
		//them from the y_providers arrays
		for (int i = 0; i < y_providers_count; i++) {

			if (out_array[i] == 1){

				int value = y_providers_value[i];
				int weight = y_providers_weight[i];

				current_slack_A -= value;
				current_slack_B += weight;

				if (current_slack_A < 0){

					//if we are here, we have a problem
					std::cout << "We have a problem with the reorder heuristic, current slack A is negative." << std::endl;
					std::cout << "Current slack A: " << current_slack_A << std::endl;
					std::cout << "Current slack B: " << current_slack_B << std::endl;

					//print out array
					for (int j = 0; j < y_providers_count; j++) {
						std::cout << out_array[j] << " ";
					}

				}

				y_providers_processed++;

				//remove the item from the y_providers arrays
				y_providers_value[i] = -1;
				y_providers_weight[i] = -1;

				//update the bool
				found_y = true;

			}

		}

	}

	//if both bools are false, return false,
	//otherwise we update the new slack values
	//and return true
	if (!found_x && !found_y)
		return false;
	
	else {

		/*if (pure_providers_count == 0 && slack_A == 0 && slack_B == 0) {

			std::cout << "We somehow found a solution to the reorder problem without any providers (this is not possible)." << std::endl;
			std::cout << "Set that caused this: " << std::endl;

			for (int i = 0; i < num_items; i++) {

				if (current_solution[i] != -1)
					std::cout << "Item " << i + 1 << " with cores: " << task_table[i * 2] << " and sms: " << task_table[(i * 2) + 1] << std::endl;

			}

			exit(1);

		}*/

		*slack_A = current_slack_A;
		*slack_B = current_slack_B;

		return true;

	}

}

HOST_DEVICE_GLOBAL void device_do_schedule(int num_tasks, int maxCPU, int NUMGPUS, int* task_table, double* losses, double* final_loss, int* uncooperative_tasks, int* final_solution, int slack_A, int slack_B, int constricted){

	//shared variables for determining the start and end of 
	//the indices for uncooperative tasks
	#ifdef __NVCC__

		// Declare shared memory pointer
		extern __shared__ char shared_mem[];
		
		// Calculate offsets for proper alignment
		size_t char_align = sizeof(char) > sizeof(void*) ? sizeof(char) : sizeof(void*);
		size_t int_align = sizeof(int) > sizeof(void*) ? sizeof(int) : sizeof(void*);
		
		// Set up base pointers
		float* float_mem = (float*)shared_mem;
		
		// Calculate offset for char memory, ensuring proper alignment
		size_t offset = sizeof(float) * 2 * (64 + 1) * (64 + 1);
		offset = (offset + char_align - 1) & ~(char_align - 1);
		char* char_mem = (char*)(shared_mem + offset);
		
		// Calculate offset for int variables
		offset += sizeof(char) * 2 * (64 + 1) * (64 + 1) * 2;
		offset = (offset + int_align - 1) & ~(int_align - 1);
		
		// Create references with the EXACT same names as the original static variables
		auto& shared_dp_two = *reinterpret_cast<float (*)[2][64 + 1][64 + 1]>(float_mem);
		auto& free_resource_pool = *reinterpret_cast<char (*)[2][64 + 1][64 + 1][2]>(char_mem);

	#else 

		float shared_dp_two[2][64 + 1][64 + 1];
		char free_resource_pool[2][64 + 1][64 + 1][2];

	#endif

	//assume 1 block of 1024 threads for now
	const int pass_count = ceil(((maxCPU + 1) * (NUMGPUS + 1)) / HOST_DEVICE_BLOCK_DIM) + 1;

	//store the indices we will be using
	#ifdef __NVCC__

		int indices[12][2];

	#endif

	//loopback variables to ensure we process
	//the uncooperative tasks last every time
	int loopback_indices[MAXTASKS];
	int loopback_back = num_tasks - 1;
	int loopback_front = 0;

	//initialize the loopback variables
	for (int i = 1; i <= num_tasks; i++){

		if (uncooperative_tasks[i - 1] != -1)
			loopback_indices[loopback_back--] = i;
		else
			loopback_indices[loopback_front++] = i;

	}

	//loop over all tasks
	for (int i = 1; i <= (int) num_tasks; i++) {

		int group_idx = loopback_indices[i - 1];

		#ifdef __NVCC__

			__syncthreads();

		#endif

		//gather task info
		int j_start = 0;
		int j_end = MAXMODES;

		//check if it is cooperative
		int desired_state = -1;

		if (uncooperative_tasks[group_idx - 1] != -1)
			desired_state = uncooperative_tasks[group_idx - 1];
	
		//for each pass we are supposed to do
		for (int k = 0; k < pass_count; k++){

			#ifdef __NVCC__ 

				__syncthreads();

				if (i == 1){

					//w = cpu
					indices[k][0] = (((k * HOST_DEVICE_BLOCK_DIM) + HOST_DEVICE_THREAD_DIM) / (NUMGPUS + 1));
					
					//v = gpu
					indices[k][1] = (((k * HOST_DEVICE_BLOCK_DIM) + HOST_DEVICE_THREAD_DIM) % (NUMGPUS + 1));

				}

				//w = cpu
				int w = indices[k][0];

				//v = gpu
				int v = indices[k][1];

			#else

				//w = cpu
				int w = (((k * HOST_DEVICE_BLOCK_DIM) + HOST_DEVICE_THREAD_DIM) / (NUMGPUS + 1));
				
				//v = gpu
				int v = (((k * HOST_DEVICE_BLOCK_DIM) + HOST_DEVICE_THREAD_DIM) % (NUMGPUS + 1));

			#endif

			if (w > maxCPU || v > NUMGPUS)
				continue;

			//invalid state
			float best_loss = 100000;
			int best_item = -1;

			//free resource pool candiates
			int best_free_cores = 0;
			int best_free_sms = 0;

			//for each item in class
			for (int j = j_start; j < j_end; j++) {

				//fetch initial suspected resource values
				int current_item_sms = constant_task_table[(group_idx - 1) * MAXMODES * 3 + j * 3 + 1];
				int current_item_cores = constant_task_table[(group_idx - 1) * MAXMODES * 3 + j * 3];

				int current_item_real_mode = constant_task_table[(group_idx - 1) * MAXMODES * 3 + j * 3 + 2];

				if (desired_state != -1)
					if (current_item_real_mode != desired_state){
						continue;
					}

				//check the change in processors
				int delta_cores = task_table[(group_idx - 1) * 2] - current_item_cores;
				int delta_sms = task_table[((group_idx - 1) * 2) + 1] - current_item_sms;

				if (current_item_cores == -1 || current_item_sms == -1)
					continue;

				//if item fits in both sacks
				if ((w < current_item_cores) || (v < current_item_sms))
					continue;

				float dp_table_loss = shared_dp_two[(i - 1) & 1][w - current_item_cores][v - current_item_sms];

				//fetch the free cores and sms
				int free_cores = free_resource_pool[(i - 1) & 1][w - current_item_cores][v - current_item_sms][0];
				int free_sms = free_resource_pool[(i - 1) & 1][w - current_item_cores][v - current_item_sms][1];

				//if we are on first pass, table is inaccurate
				if (i == 1){

					dp_table_loss = 0;

					free_cores = slack_A;
					free_sms = slack_B;

				}

				//check if our resource constrains are maintained
				if ((dp_table_loss != 100000) && (delta_cores * delta_sms < 0)){

					bool reorder = false;

					//if cores is negative, make sure we have enough
					//free cores to cover it
					if (delta_cores < 0){

						if (free_cores + delta_cores < 0){
						
							reorder = true;

						}

					}

					//if sms is negative, make sure we have enough
					//free sms to cover it
					if (delta_sms < 0){

						if (free_sms + delta_sms < 0){
						
							reorder = true;

						}

					}

					//if we need to reorder, we run the new heuristic
					if (reorder){

						//fetch the solution
						int reordered_solution[MAXTASKS];
						int correctly_ordered[MAXTASKS];

						for (int l = 0; l < MAXTASKS; l++){
							reordered_solution[l] = -1;
							correctly_ordered[l] = -1;
						}
						
						//when the soluion is rebuilt, we have to associate
						//the items with the loopback indices correctly
						rebuild_solution(reordered_solution, loopback_indices, i - 1, w - current_item_cores, v - current_item_sms);

						for (int l = 0; l < (i - 1); l++)
							correctly_ordered[loopback_indices[l] - 1] = reordered_solution[l];

						for (int l = 0; l < MAXTASKS; l++)
							reordered_solution[l] = correctly_ordered[l];

						reordered_solution[group_idx - 1] = j;

						int old_slack_A = free_cores;
						int old_slack_B = free_sms;

						//now we need to reorder the solution
						if (!reorder_heuristic(reordered_solution, task_table, num_tasks, &free_cores, &free_sms, 1))
							if (!reorder_heuristic(reordered_solution, task_table, num_tasks, &free_cores, &free_sms, 0))
								continue;

						if ((i > 4) && (old_slack_A != free_cores || old_slack_B != free_sms)){

							std::cout << "free cores:" << free_cores << std::endl;
							std::cout << "free sms:" << free_sms << std::endl;

							std::cout << "Target Selected Item: " << solutions[i-1][w - current_item_cores][v - current_item_sms] << std::endl;

							std::cout << "Reordering solution for task " << group_idx << " with item " << j << std::endl;
							std::cout << "Current solution: " << std::endl;
							for (int l = 0; l < MAXTASKS; l++){

									std::cout << "Item " << l + 1 << ": " << reordered_solution[l] << std::endl;

							}

							std::cout << "groupidx order : " << std::endl;
							for (int l = 0; l < i; l++){

								if (loopback_indices[l] != -1)
									std::cout << "Item " << l + 1 << ": " << loopback_indices[l] << std::endl;

							}

							//print the new slack
							std::cout << "New slack A: " << free_cores << std::endl;
							std::cout << "New slack B: " << free_sms << std::endl;

							exit(1);

						}
						
						free_cores -= delta_cores;
						free_sms -= delta_sms;
						

					}

				}

				if ((dp_table_loss != 100000)) {

					float newCPULoss_two = dp_table_loss + constant_losses[(group_idx - 1) * MAXMODES + j];
					
					//if found solution is better, update
					if ((newCPULoss_two) < (best_loss)) {

						best_loss = newCPULoss_two;

						best_item = j;

						best_free_cores = free_cores + delta_cores;

						best_free_sms = free_sms + delta_sms;

					}

				}

			}

			//store the best loss
			shared_dp_two[i & 1][w][v] = best_loss;

			//store the best item
			solutions[i][w][v] = best_item;

			//store the best free cores and sms
			free_resource_pool[i & 1][w][v][0] = best_free_cores;
			free_resource_pool[i & 1][w][v][1] = best_free_sms;

		}

	}

	#ifdef __NVCC__

		__syncthreads();

	#endif

	//to get the final answer, start at the end and work backwards, taking the j values
	if (HOST_DEVICE_THREAD_DIM < 1){

		//rebuild the solution
		bool valid_solution = rebuild_solution(final_solution, loopback_indices, num_tasks, maxCPU, NUMGPUS);

		//print the final loss 
		if (valid_solution){

			*final_loss = shared_dp_two[num_tasks & 1][maxCPU][NUMGPUS];

		} else {

			*final_loss = 100001;

		}

	}

}