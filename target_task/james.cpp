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
int iterations_complete = 0;

extern int task_index;


#ifdef __NVCC__

   #include "libsmctrl.h"
   #include <cuda.h>
   #include <cuda_runtime.h>

   #include "sm_mapper.cuh"

   cudaStream_t stream;

   bool display_sms = true;

#endif

void update_core_B(__uint128_t mask) {

   std::ostringstream buffer;

   print_module::buffered_print(buffer, task_index, " -> Core B Mask: ");

   //print the mask
   for (int i = 0; i < 128; i++) {
       print_module::buffered_print(buffer, (unsigned long long)((mask & ((__uint128_t)1 << (__uint128_t(i)))) >> (__uint128_t(i))));
   }

   print_module::buffered_print(buffer, "\n");
   print_module::flush(std::cerr, buffer);

    //example of how to use core B masks
   #ifdef __NVCC__

       libsmctrl_set_stream_mask(stream, ~mask);

       //if first time, print sms
        if (display_sms) {

            visualize_sm_partitions_interprocess(stream, 3, "JAMESSM");
            display_sms = false;
            
        }

   #endif

}

int init(int argc, char *argv[])
{

   #ifdef __NVCC__

       cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

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

    if (task_index == 2)
        set_cooperative(false);

   return 0;       
}

int run(int argc, char *argv[]){

    std::atomic<int> count = 0;

    #pragma omp parallel
    {

        pm::task_print(std::cerr, "Thread ", omp_get_thread_num(), " on core ", sched_getcpu(), " of ", omp_get_num_threads(), " threads\n");

        count++;

        busy_work(spin_tv);
        
    }

    std::cout << "TEST: [" << task_index << "," << iterations_complete << "] core count: " << count << std::endl;

    pm::task_print(std::cout, "TEST: [", task_index, ",", iterations_complete, "] core count: ", count, "\n");

    iterations_complete++;

    if (task_index == 2 && iterations_complete % 5 == 0 && iterations_complete % 2 == 1) {
        synth_current_mode = (synth_current_mode + 1) % mode_count;
        modify_self(1);
    }

    if (task_index == 2 && iterations_complete % 5 == 0 && iterations_complete % 2 == 0) {
        synth_current_mode = (synth_current_mode + 1) % mode_count;
        modify_self(3);
    }

    return 0;
}

int finalize(int argc, char *argv[])
{
   return 0;
}

task_t task = { init, run, finalize, update_core_B };
