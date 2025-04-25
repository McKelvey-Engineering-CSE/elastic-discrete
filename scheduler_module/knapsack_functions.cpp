#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __NVCC__

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

#else 

	#define HOST_DEVICE_BLOCK_DIM 1
	#define HOST_DEVICE_THREAD_DIM 1
	#define HOST_DEVICE_SCOPE
	#define HOST_DEVICE_CONSTANT
	#define HOST_DEVICE_GLOBAL static

#endif

__shared__ float shared_dp_two[2][64 + 1][64 + 1];

HOST_DEVICE_SCOPE volatile int solutions[MAXTASKS][128 + 1][128 + 1][2];

HOST_DEVICE_CONSTANT int constant_task_table[MAXTASKS * MAXMODES * 3];

HOST_DEVICE_CONSTANT double constant_losses[MAXTASKS * MAXMODES];

HOST_DEVICE_GLOBAL void device_do_schedule(int num_tasks, int maxCPU, int NUMGPUS, int* task_table, double* losses, double* final_loss, int* uncooperative_tasks, int* final_solution){

	//shared variables for determining the start and end of 
	//the indices for uncooperative tasks
	#ifdef __NVCC__

		__shared__ int start_index;
		__shared__ int end_index;

	#endif

	int modes_skipped = 0;

	//assume 1 block of 1024 threads for now
	const int pass_count = ceil(((maxCPU + 1) * (NUMGPUS + 1)) / HOST_DEVICE_BLOCK_DIM) + 1;

	//store the indices we will be using
	int indices[12][2];

	//loop over all tasks
	for (int i = 1; i <= (int) num_tasks; i++) {

		//gather task info
		int j_start = 0;
		int j_end = MAXMODES;

		//check if it is cooperative

		#ifdef __NVCC__
		
		if (uncooperative_tasks[i - 1]){

			if (threadIdx.x < MAXMODES - 1){

				//read one element from the target task modes which is our index
				int our_element = constant_task_table[(i - 1) * MAXMODES * 3 + threadIdx.x * 3 + 2];

				//read the next element too
				int next_element = constant_task_table[(i - 1) * MAXMODES * 3 + (threadIdx.x + 1) * 3 + 2];

				//if our element is the uncooperative task target mode, and the next element is not, we have j_end. 
				if (our_element == uncooperative_tasks[i - 1] && next_element != uncooperative_tasks[i - 1]){

					//store the end index
					atomicExch(&end_index, threadIdx.x + 1);

				}

				//if our element is not the uncooperative task target mode, and the next element is, we have j_start
				if (our_element != uncooperative_tasks[i - 1] && next_element == uncooperative_tasks[i - 1]){

					//store the start index
					atomicExch(&start_index, threadIdx.x);

				}

			}

			__syncthreads();

			//update the start and end indices
			j_start = start_index;
			j_end = end_index;

		}

		#else 

			int desired_state = -1;
			if (uncooperative_tasks[i - 1])
				desired_state = uncooperative_tasks[i - 1];
		
		#endif	

		//for each pass we are supposed to do
		for (int k = 0; k < pass_count; k++){

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

			if (w > maxCPU || v > NUMGPUS)
				continue;

			//invalid state
			shared_dp_two[(i & 1)][w][v] = -1.0;

			//for each item in class
			for (size_t j = j_start; j < j_end; j++) {

				#ifndef __NVCC__

					if (desired_state != -1)
						if (current_item_real_mode != desired_state)
							continue;
				#endif

				//fetch initial suspected resource values
				int current_item_sms = constant_task_table[(i - 1) * MAXMODES * 3 + j * 3 + 1];
				int current_item_cores = constant_task_table[(i - 1) * MAXMODES * 3 + j * 3];
				int current_item_real_mode = constant_task_table[(i - 1) * MAXMODES * 3 + j * 3 + 2];

				//check the change in processors
				int delta_cores = task_table[(i - 1) * 2] - current_item_cores;
				int delta_sms = task_table[((i - 1) * 2) + 1] - current_item_sms;

				if (delta_cores * delta_sms < 0)
					continue;

				if (current_item_cores == -1 || current_item_sms == -1)
					continue;

				//if item fits in both sacks
				if ((w < current_item_cores) || (v < current_item_sms))
					continue;

				float dp_table_loss = shared_dp_two[(i - 1) & 1][w - current_item_cores][v - current_item_sms];

				if (i == 1)
					dp_table_loss = 100000;

				if ((dp_table_loss != -1)) {

					float newCPULoss_two = dp_table_loss - (float) constant_losses[(i - 1) * MAXMODES + j];
					
					//if found solution is better, update
					if ((newCPULoss_two) > (shared_dp_two[i & 1][w][v])) {

						shared_dp_two[i & 1][w][v] = newCPULoss_two;

						//store j into the corresponding slot of the 1d array in the first position
						solutions[i][w][v][0] = j;

						//store a pointer to the previous portion of the solution in the second position
						solutions[i][w][v][1] = ((unsigned)(i - 1) << 16) | ((unsigned)(w - current_item_cores) << 8) | (unsigned)(v - current_item_sms);

					}

				}

			}

			#ifdef __NVCC__ 

				__syncthreads();

			#endif

		}

	}

	//to get the final answer, start at the end and work backwards, taking the j values
	if (HOST_DEVICE_THREAD_DIM < 1){

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
		*final_loss = 100000 - shared_dp_two[num_tasks & 1][maxCPU][NUMGPUS];

	}

}