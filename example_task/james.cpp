/*
Configurable task to use in unit tests. Features:
* Takes a mode count as an argument, cycles bewteen the modes at a configurable interval
* Does a configurable amount of work in the meantime
* Prints the number of cores on which it is running at each iteration
*/

#include "task.h"
#include <cstring>
#include <omp.h>
#include "timespec_functions.h"

int logging_index = -1;

timespec spin_tv;
int mode_count = 0;
int mode_change_interval = -1;

int synth_current_mode = 0;
int iterations_complete = -1;

extern int task_index;

bool first_time = true;

#ifdef __NVCC__

    #include <cuda.h>
    #include <cuda_runtime.h>

    #include "sm_mapper.cuh"
 
    cudaStream_t stream;

    CUgreenCtx task_green_ctx;

    CUdevResource resources[128];

    bool display_sms = true;

#endif

//#define LOUD_PRINT
//#define SM_PRINT

void update_core_B(__uint128_t mask) {

    //for a hybrid CPU, omp_replacement still controls both the A and B cores
    //but you must update the mask to reflect the B cores the task owns as well
    omp.set_override_mask(processor_A_mask | (processor_B_mask >> NUM_PROCESSOR_A));

    return;

}

void update_core_C(__uint128_t mask) {

    //for now just pretend
    return;

    //example of how to use an additional core mask
    //for something like a GPU, in this case we are 
    //using green contexts to control the GPU TPCs
    #ifdef __NVCC__

        //device specs
        CUdevResourceDesc device_resource_descriptor;
        CUdevResource initial_resources;
        unsigned int partition_num;

        //fill the initial descriptor
        cuDeviceGetDevResource(0, &initial_resources, CU_DEV_RESOURCE_TYPE_SM);

        partition_num = initial_resources.sm.smCount / 2;

        //take the previous element above us and split it 
        //fill the corresponding portions of the matrix as we go
        CUDA_SAFE_CALL(cuDevSmResourceSplitByCount(resources, &partition_num, &initial_resources, NULL, CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));

        //now copy the TPC elements we have been granted
        unsigned int total_TPCS = __builtin_popcount(mask);

        std::cerr << "Total TPCS: " << total_TPCS << std::endl;

        if (total_TPCS == 0)
            return;

        CUdevResource my_resources[total_TPCS];
        int next = 0;

        for (int i = 0; i < 128; i++){

            if (mask & ((__uint128_t)1 << i)) {

                my_resources[next++] = resources[i];

            }

        }

        //now make a green context from all the other resources
        CUDA_SAFE_CALL(cuDevResourceGenerateDesc(&device_resource_descriptor, my_resources, total_TPCS));
        CUDA_SAFE_CALL(cuGreenCtxCreate(&task_green_ctx, device_resource_descriptor, 0, CU_GREEN_CTX_DEFAULT_STREAM));

        //make a stream as well
        CUDA_SAFE_CALL(cuGreenCtxStreamCreate(&stream, task_green_ctx, CU_STREAM_NON_BLOCKING, 0));

    //if first time, print sms
    #ifdef SM_PRINT
    
        if (display_sms) {

            visualize_sm_partitions_interprocess(task_green_ctx, 1, "JAMESSM");
            display_sms = false;
            
        }

    #endif

    #endif

}

void update_core_D(__uint128_t mask) {

    //for now just pretend
    return;

    //example of how to use core D masks
    //Similar to update_core_B but for processor type D
}

int init(int argc, char *argv[])
{

   #ifdef __NVCC__

       CUDA_SAFE_CALL(cuInit(0));

   #endif

   if (argc < 2) {
       std::cerr << "synthetic_test_task: not enough arguments" << std::endl;
       return -1;
   }

   spin_tv.tv_sec = 0;

   //TODO all args are passed in as a single string - should change clustering_launcher to fix this
   if (sscanf(argv[1], "%d %ld %d %d", &logging_index, &spin_tv.tv_nsec, &mode_count, &mode_change_interval) < 3) {
       std::cerr << "synthetic_test_task: failed to parse args" << std::endl;
       return -2;
   }

   return 0;       
}

int run(int argc, char *argv[]){

    std::ostringstream buffer;

    std::atomic<int> count = 0;

    #ifdef LOUD_PRINT
        print_module::buffered_print(buffer, "\n(", getpid(), ") [", task_index, "] [Threads]: \n");
    #endif

    #ifdef OMP_OVERRIDE

        omp( pragma_omp_parallel
        {
            #ifdef LOUD_PRINT
                pm::buffered_print(buffer, "ompish Thread ", thread_id, " on core ", sched_getcpu(), " of ", team_dim, " threads\n");
            #endif

            count++;

            busy_work(spin_tv);
            
        });

    #else

        #pragma omp parallel
        {

            #ifdef LOUD_PRINT
                pm::buffered_print(buffer, "omp Thread ", omp_get_thread_num(), " on core ", sched_getcpu(), " of ", omp_get_num_threads(), " threads\n");
            #endif
            
            count++;

            busy_work(spin_tv);
            
        }

    #endif

    #ifdef LOUD_PRINT
        pm::buffered_print(buffer, "TEST: [", task_index, ",", iterations_complete, "] core count: ", count, "\n");
        pm::flush(std::cerr, buffer);
    #endif

    const int instigation_time[] = {3, 5, 7, 11, 13};

    if (task_index < 5){

        if (iterations_complete % instigation_time[task_index] == 0){

            mimic_simulator(task_index);

        }

    }

    iterations_complete++;

    return 0;
}

int finalize(int argc, char *argv[])
{
   return 0;
}

task_t task = { init, run, finalize, update_core_B, update_core_C, update_core_D };
