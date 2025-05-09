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

HOST_DEVICE_GLOBAL void device_do_schedule(int num_tasks, int maxCPU, int NUMGPUS, int* task_table, double* losses, double* final_loss, int* uncooperative_tasks, int* final_solution, int slack_A, int slack_B){

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
		int* int_mem = (int*)(shared_mem + offset);
		
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
	int indices[12][2];

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
				if (delta_cores * delta_sms < 0){

					//if cores is negative, make sure we have enough
					//free cores to cover it
					if (delta_cores < 0){

						if (free_cores + delta_cores < 0)
							continue;

					}

					//if sms is negative, make sure we have enough
					//free sms to cover it
					if (delta_sms < 0){

						if (free_sms + delta_sms < 0)
							continue;

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

		//print the final loss 
		if (valid_solution){

			*final_loss = shared_dp_two[num_tasks & 1][maxCPU][NUMGPUS];

		} else {

			*final_loss = 100001;

		}

	}

}