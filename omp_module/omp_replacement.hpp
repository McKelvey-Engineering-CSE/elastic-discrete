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
        pthread_t* threads;
        int num_threads;

    public:
        OMPThreadPool(int num_threads){

            //make array for pthread handles
            this->threads = new pthread_t[num_threads];
            this->num_threads = num_threads;

            //set omp num threads
            omp_set_num_threads(num_threads);

            //make each thread fetch their handle
            #pragma omp parallel
            {
                threads[omp_get_thread_num()] = pthread_self();
            }

        }

        ~OMPThreadPool(){
            delete[] threads;
        }

        void set_thread_pool_affinity(__uint128_t affinity_mask){

            //translate the mask into a vector
            std::vector<int> cores;

            for (int i = 0; i < 128; i++) {
                if (affinity_mask & ((__uint128_t)1 << i)) {
                    cores.push_back(i);
                }
            }

            if (cores.size() == 0) {
                //no cores available
                return;
            }

            //set thread affinity for all threads (1 core to each thread in a circle buffer)
            for (int i = 0; i < num_threads; i++) {
                
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(cores[i % cores.size()], &cpuset);
        
                if (pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset) != 0) {
                    perror("pthread_setaffinity_np");
                }
            }

        }
};

#endif