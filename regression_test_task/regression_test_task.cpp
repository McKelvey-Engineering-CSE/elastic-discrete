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

int core_b_count = 0;

void update_core_B(__uint128_t mask) {
    int count = 0;
    std::bitset<128> thread_mask(mask);
    for (int i = 0; i < 128; i++) {
        if (thread_mask[i]) {
            count++;
        }
    }
    core_b_count = count;
}

int init(int argc, char *argv[])
{

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
    auto current_mask = omp.get_override_mask();

   std::bitset<128> thread_mask(current_mask);
    int resource_a_count = 0;
    for (size_t i = 1; i < 128; i++) {
        if (thread_mask[i]) {
            resource_a_count++;
        }
    }

    std::cout << "TEST: [" << logging_index << "," << iterations_complete << "] core count: A: " << resource_a_count << ", B: " << core_b_count << std::endl;

    iterations_complete++;

    if (mode_count > 1 && mode_change_interval > 0 && iterations_complete > 0 && (iterations_complete % mode_change_interval == 0)) {
        synth_current_mode = (synth_current_mode + 1) % mode_count;
        modify_self(synth_current_mode);
    }

	return 0;
}

int finalize(int argc, char *argv[])
{
    return 0;
}

task_t task = { init, run, finalize, update_core_B };
