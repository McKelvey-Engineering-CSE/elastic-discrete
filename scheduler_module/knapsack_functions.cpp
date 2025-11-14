#include <iostream>
#include <string>

#include "include.h"

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
	#define HOST_DEVICE_THREAD_IDX threadIdx.x
	#define HOST_DEVICE_SCOPE __device__
	#define HOST_DEVICE_CONSTANT __constant__
	#define HOST_DEVICE_GLOBAL __global__
	#define HOST_DEVICE_SHARED __shared__

#else 

	#define HOST_DEVICE_BLOCK_DIM 1
	#define HOST_DEVICE_THREAD_IDX 0
	#define HOST_DEVICE_SCOPE
	#define HOST_DEVICE_CONSTANT
	#define HOST_DEVICE_GLOBAL static
	#define HOST_DEVICE_SHARED static

#endif

HOST_DEVICE_SCOPE volatile int solutions[MAXTASKS][NUM_PROCESSOR_A + 1][NUM_PROCESSOR_B + 1][NUM_PROCESSOR_C + 1][NUM_PROCESSOR_D + 1];

HOST_DEVICE_CONSTANT int constant_task_table[MAXTASKS * MAXMODES * 5];

HOST_DEVICE_CONSTANT float constant_losses[MAXTASKS * MAXMODES];

HOST_DEVICE_SCOPE float shared_dp_two[2][NUM_PROCESSOR_A + 1][NUM_PROCESSOR_B + 1][NUM_PROCESSOR_C + 1][NUM_PROCESSOR_D + 1];

HOST_DEVICE_SCOPE char free_resource_pool[2][NUM_PROCESSOR_A + 1][NUM_PROCESSOR_B + 1][NUM_PROCESSOR_C + 1][NUM_PROCESSOR_D + 1][4];

HOST_DEVICE_GLOBAL void device_do_schedule(int num_tasks, int* task_table, double* losses, double* final_loss, int* uncooperative_tasks, int* final_solution, int slack_A, int slack_B, int slack_C, int slack_D, int constricted, int max_A, int max_B, int max_C, int max_D){

	//shared variables for determining the start and end of 
	//the indices for uncooperative tasks
	/*#ifdef __NVCC__

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

	#endif*/

	// Clamp max bounds to not exceed compile-time array bounds
	const int clamped_max_A = (max_A < 0 || max_A > NUM_PROCESSOR_A) ? NUM_PROCESSOR_A : max_A;
	const int clamped_max_B = (max_B < 0 || max_B > NUM_PROCESSOR_B) ? NUM_PROCESSOR_B : max_B;
	const int clamped_max_C = (max_C < 0 || max_C > NUM_PROCESSOR_C) ? NUM_PROCESSOR_C : max_C;
	const int clamped_max_D = (max_D < 0 || max_D > NUM_PROCESSOR_D) ? NUM_PROCESSOR_D : max_D;
	
	//assume 1 block of 1024 threads for now
	const int pass_count = ceil(((clamped_max_A + 1) * (clamped_max_B + 1) * (clamped_max_C + 1) * (clamped_max_D + 1)) / HOST_DEVICE_BLOCK_DIM) + 1;

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

			#endif

			// Map 1D thread index to 4D coordinates for processors A, B, C, D
			int thread_id = (k * HOST_DEVICE_BLOCK_DIM) + HOST_DEVICE_THREAD_IDX;
			
			//a = processor A
			int a = (thread_id / ((clamped_max_B + 1) * (clamped_max_C + 1) * (clamped_max_D + 1))) % (clamped_max_A + 1);
			
			//b = processor B
			int b = (thread_id / ((clamped_max_C + 1) * (clamped_max_D + 1))) % (clamped_max_B + 1);
			
			//c = processor C
			int c = (thread_id / (clamped_max_D + 1)) % (clamped_max_C + 1);
			
			//d = processor D
			int d = thread_id % (clamped_max_D + 1);

			if (a > clamped_max_A || b > clamped_max_B || c > clamped_max_C || d > clamped_max_D)
				continue;

			//invalid state
			float best_loss = 100000;
			int best_item = -1;

			//free resource pool candidates for all processor types
			int best_free_cores_A = 0;
			int best_free_cores_B = 0;
			int best_free_cores_C = 0;
			int best_free_cores_D = 0;

			//for each item in class
			for (int j = j_start; j < j_end; j++) {

				//fetch initial suspected resource values
				int current_item_cores_A = constant_task_table[(group_idx - 1) * MAXMODES * 5 + j * 5 + 0];
				int current_item_cores_B = constant_task_table[(group_idx - 1) * MAXMODES * 5 + j * 5 + 1];
				int current_item_cores_C = constant_task_table[(group_idx - 1) * MAXMODES * 5 + j * 5 + 2];
				int current_item_cores_D = constant_task_table[(group_idx - 1) * MAXMODES * 5 + j * 5 + 3];
				int current_item_real_mode = constant_task_table[(group_idx - 1) * MAXMODES * 5 + j * 5 + 4];

				if (desired_state != -1){
					   
					//this means we have a desired state we must stay in
					if (desired_state > -1)
						if (current_item_real_mode != desired_state)
							continue;

					//this means we have a desired state that we must not go below
					else {
						if (current_item_real_mode < (desired_state * -1) - 1)
							continue;
					}

				}

				//check the change in processors
				int delta_cores_A = task_table[(group_idx - 1) * 4 + 0] - current_item_cores_A;
				int delta_cores_B = task_table[(group_idx - 1) * 4 + 1] - current_item_cores_B;
				int delta_cores_C = task_table[(group_idx - 1) * 4 + 2] - current_item_cores_C;
				int delta_cores_D = task_table[(group_idx - 1) * 4 + 3] - current_item_cores_D;

				if (current_item_cores_A == -1 || current_item_cores_B == -1 || current_item_cores_C == -1 || current_item_cores_D == -1)
					continue;

				//if item fits in all four sacks
				if ((a < current_item_cores_A) || (b < current_item_cores_B) || (c < current_item_cores_C) || (d < current_item_cores_D))
					continue;

				float dp_table_loss = shared_dp_two[(i - 1) & 1][a - current_item_cores_A][b - current_item_cores_B][c - current_item_cores_C][d - current_item_cores_D];

				//fetch the free cores for all processor types
				int free_cores_A = free_resource_pool[(i - 1) & 1][a - current_item_cores_A][b - current_item_cores_B][c - current_item_cores_C][d - current_item_cores_D][0];
				int free_cores_B = free_resource_pool[(i - 1) & 1][a - current_item_cores_A][b - current_item_cores_B][c - current_item_cores_C][d - current_item_cores_D][1];
				int free_cores_C = free_resource_pool[(i - 1) & 1][a - current_item_cores_A][b - current_item_cores_B][c - current_item_cores_C][d - current_item_cores_D][2];
				int free_cores_D = free_resource_pool[(i - 1) & 1][a - current_item_cores_A][b - current_item_cores_B][c - current_item_cores_C][d - current_item_cores_D][3];

				//if we are on first pass, table is inaccurate
				if (i == 1){

					dp_table_loss = 0;

					free_cores_A = slack_A;
					free_cores_B = slack_B;
					free_cores_C = slack_C;
					free_cores_D = slack_D;

				}

				//check if our resource constrains are maintained
				if (delta_cores_A < 0 || delta_cores_B < 0 || delta_cores_C < 0 || delta_cores_D < 0){

					if (constricted == 0){

						//if any processor type is negative, make sure we have enough
						//free resources to cover it
						if (free_cores_A + delta_cores_A < 0 || 
							free_cores_B + delta_cores_B < 0 || 
							free_cores_C + delta_cores_C < 0 || 
							free_cores_D + delta_cores_D < 0) {
							continue;
						}
					
					}

					else {

						continue;
					
					}

				}

				if ((dp_table_loss != 100000)) {

					float newCPULoss_two = dp_table_loss + constant_losses[(group_idx - 1) * MAXMODES + j];
					
					//if found solution is better, update
					if ((newCPULoss_two) < (best_loss)) {

						best_loss = newCPULoss_two;

						best_item = j;

						best_free_cores_A = free_cores_A + delta_cores_A;
						best_free_cores_B = free_cores_B + delta_cores_B;
						best_free_cores_C = free_cores_C + delta_cores_C;
						best_free_cores_D = free_cores_D + delta_cores_D;

					}

				}

			}

			//store the best loss
			shared_dp_two[i & 1][a][b][c][d] = best_loss;

			//store the best item
			solutions[i][a][b][c][d] = best_item;

			//store the best free cores for all processor types
			free_resource_pool[i & 1][a][b][c][d][0] = best_free_cores_A;
			free_resource_pool[i & 1][a][b][c][d][1] = best_free_cores_B;
			free_resource_pool[i & 1][a][b][c][d][2] = best_free_cores_C;
			free_resource_pool[i & 1][a][b][c][d][3] = best_free_cores_D;

		}

	}

	#ifdef __NVCC__

		__syncthreads();

	#endif

	//to get the final answer, start at the end and work backwards, taking the j values
	if (HOST_DEVICE_THREAD_IDX < 1){

		bool valid_solution = true;
		int current_a = clamped_max_A - 1;
		int current_b = clamped_max_B;
		int current_c = clamped_max_C;
		int current_d = clamped_max_D;

		for (int i = num_tasks; i > 0; i--){

			int group_idx = loopback_indices[i - 1];

			int current_item = solutions[i][current_a][current_b][current_c][current_d];

			if (current_item == -1){

				valid_solution = false;

			}

			else {

				//take the core values for all processor types for the item
				int current_item_cores_A = constant_task_table[(group_idx - 1) * MAXMODES * 5 + current_item * 5 + 0];
				int current_item_cores_B = constant_task_table[(group_idx - 1) * MAXMODES * 5 + current_item * 5 + 1];
				int current_item_cores_C = constant_task_table[(group_idx - 1) * MAXMODES * 5 + current_item * 5 + 2];
				int current_item_cores_D = constant_task_table[(group_idx - 1) * MAXMODES * 5 + current_item * 5 + 3];

				//update the current a, b, c, d values
				current_a = current_a - current_item_cores_A;
				current_b = current_b - current_item_cores_B;
				current_c = current_c - current_item_cores_C;
				current_d = current_d - current_item_cores_D;
				
			}

			final_solution[i - 1] = current_item;

		}

		//print the final loss 
		if (valid_solution){

			*final_loss = shared_dp_two[num_tasks & 1][clamped_max_A - 1][clamped_max_B][clamped_max_C][clamped_max_D];

		} else {

			*final_loss = 100001;

		}

	}

}