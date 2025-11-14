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

class OMPThreadPool {

    private:
        int num_threads;

    public:
        OMPThreadPool(int _num_threads) : num_threads(_num_threads) {

            //set omp num threads
            omp_set_num_threads(_num_threads);

            //make each thread fetch their handle
            #pragma omp parallel
            {

                //make each thread set their scheduling policy to SCHED_FIFO with priority 90
                struct sched_param param;
                param.sched_priority = 90;
                pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

            }

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

            //set the omp thread count
            omp_set_num_threads(current_num_threads);

            //translate the mask into a vector
            int cores[current_num_threads];

            //get all the set bits in the mask
            get_set_bits(affinity_mask, cores);

            //set thread affinity for all threads (1 core to each thread in a circle buffer)
            #pragma omp parallel
            {

                if (omp_get_thread_num() != 0){

                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(cores[omp_get_thread_num()], &cpuset);
                    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                
                }

            }

        }
};

#endif