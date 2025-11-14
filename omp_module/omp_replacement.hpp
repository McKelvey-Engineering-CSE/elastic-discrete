#ifndef OMP_REPLACEMENT_HPP
#define OMP_REPLACEMENT_HPP

#include <vector>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <bitset>
#include <cassert>
#include <omp.h>
#include <iostream>

class OMPThreadPool {

    private:
        int num_threads;

    public:
        OMPThreadPool(int _num_threads) : num_threads(_num_threads) {

        }

        ~OMPThreadPool(){
        }

        void get_set_bits(unsigned long long n, int* indices) {
            int count = 0;
            while (n) {
                int index = __builtin_ctzll(n);
                indices[count++] = index;
                n &= n - 1;
            }
        }

        void set_thread_pool_affinity(__uint128_t affinity_mask){

            int current_num_threads = __builtin_popcount(affinity_mask);

            //kill all threads
            omp_set_num_threads(1);

            //translate the mask into a vector
            int cores[current_num_threads];

            //core we do not touch 
            int permanent_core = sched_getcpu();

            //mask without the permanent core
            auto affinity_mask_without_permanent_core = affinity_mask & ~((__uint128_t)1 << permanent_core);

            //get all the set bits in the mask
            get_set_bits(affinity_mask_without_permanent_core, cores);

            //set our priority to low non-rt
            struct sched_param param;
            param.sched_priority = 0;
            pthread_setschedparam(pthread_self(), SCHED_OTHER, &param);

            //spawn threads back up
            omp_set_num_threads(current_num_threads);

            //make each thread migrate to their core
            #pragma omp parallel
            {

                if (omp_get_thread_num() != 0){

                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(cores[omp_get_thread_num() - 1], &cpuset);
                    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

                }
                
            }

            //now make all threads set their priority to high rt
            #pragma omp parallel
            {
                struct sched_param param;
                param.sched_priority = 90;
                pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
            }

        }
};

#endif