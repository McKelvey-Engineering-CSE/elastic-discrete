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
    #include "libsmctrl.h"
 
    cudaStream_t stream;

    bool streams_initialized = false;

    bool display_sms = true;

#endif

//#define LOUD_PRINT
//#define SM_PRINT

void update_core_B(__uint128_t mask) {

    //for a hybrid CPU, omp_replacement still controls both the A and B cores
    //but you must update the mask to reflect the B cores the task owns as well
    //omp_threadpool->set_thread_pool_affinity(processor_A_mask | (processor_B_mask << NUM_PROCESSOR_A));

    return;

}

void update_core_C(__uint128_t mask) {

    //for now just pretend
    return;

    //example of how to use an additional core mask
    //for something like a GPU via libsmctrl
    #ifdef __NVCC__

    if (mask == 0){
        
        return;

    }

    //make the streams if they are not already made
    if (!streams_initialized){
    
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        streams_initialized = true;
    
    }

    libsmctrl_set_stream_mask(stream, ~mask);

    return;

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

    //int count = 0;

    #pragma omp parallel
    {

        #ifdef PRETTY_PRINTING
            std::ostringstream buffer;
            pm::buffered_print(buffer, "[", task_index, "] omp Thread ", omp_get_thread_num(), " on core ", sched_getcpu(), " of ", omp_get_num_threads(), " threads\n");
            pm::flush(std::cerr, buffer);
        #endif

        sleep_for_ts(spin_tv);
        
    }

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
