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

#define pragma_omp_parallel true, [&](int thread_rank, int team_dim, int thread_id)

template <typename T = void(int, int, int)>
class ThreadPool {
private:
    static constexpr int MAX_THREADS = 128;
    
    std::vector<std::thread> workers;
    std::function<T> task;
    std::vector<std::mutex> queue_mutex;
    std::vector<std::condition_variable> condition;
    std::vector<std::atomic<bool>> completed;  // Changed to atomic
    std::vector<int> dimension;
    std::vector<int> rank;
    std::atomic<bool> stop;
    std::atomic<int> threadIDs{1};
    std::atomic<int> threads_ready{1};
    std::atomic<int> active_thread_count{0};  // Added counter for active threads
    std::condition_variable completion_cv;     // Added for completion synchronization
    std::mutex completion_mutex;               // Mutex for completion synchronization
    size_t permanent_cpu = 0;
    __uint128_t global_override_mask = ~(__uint128_t)0;
    unsigned long long job_id = 0;

public:
    explicit ThreadPool(int threads) : 
        queue_mutex(MAX_THREADS),
        condition(MAX_THREADS),
        completed(MAX_THREADS),  // Initialize atomic vector
        dimension(MAX_THREADS),
        rank(MAX_THREADS),
        stop(false) {
        
        assert(threads <= MAX_THREADS && "Thread count exceeds maximum supported threads");
        
        // Initialize completed flags
        for (auto& c : completed) {
            c.store(true, std::memory_order_release);
        }
        
        workers.reserve(threads - 1);
        
        for (int i = 1; i < threads; i++) {
            workers.emplace_back([this]{
                const int my_index = threadIDs.fetch_add(1);
                bool first = true;
                unsigned long long last_job_id = -1;

                while (true) {
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex[my_index]);
                        
                        if (first) {
                            first = false;
                            threads_ready.fetch_add(1);
                            condition[0].notify_one();
                        } else {
                            completed[my_index].store(true, std::memory_order_release);
                            // Decrement active thread count and notify completion
                            if (active_thread_count.fetch_sub(1) == 1) {
                                std::lock_guard<std::mutex> completion_lock(completion_mutex);
                                completion_cv.notify_one();
                            }
                        }
                        
                        condition[my_index].wait(lock, [this, my_index, last_job_id] {
                            return (stop || !completed[my_index].load(std::memory_order_acquire));
                        });
                        
                        if (stop) return;
                    }
                    
                    if (last_job_id == job_id) {
                        continue;
                    }

                    task(rank[my_index], dimension[my_index], my_index);
                    last_job_id = job_id;
                }
            });
        }

        // Wait for all threads to initialize
        {
            std::unique_lock<std::mutex> lock(queue_mutex[0]);
            condition[0].wait(lock, [this, threads]{
                return threads_ready == threads;
            });
        }
    }

    template<class F>
    void execute_parallel(F&& f, __uint128_t mask) {
        std::unique_lock<std::mutex> lock(queue_mutex[0]);

        std::bitset<MAX_THREADS> thread_mask(mask);
        task = std::forward<F>(f);
        job_id += 1;

        const int thread_dim = static_cast<int>(thread_mask.count());
        int rank_ct = 0;
        bool participating = false;

        // Reset active thread count
        active_thread_count.store(0, std::memory_order_release);
        
        // Wake up the correct threads
        for (size_t i = 1; i < workers.size() + 1; ++i) {
            if (thread_mask[i]) {
                if (permanent_cpu != i) {
                    std::lock_guard<std::mutex> lock(queue_mutex[i]);
                    completed[i].store(false, std::memory_order_release);
                    dimension[i] = thread_dim;
                    rank[i] = rank_ct++;
                    active_thread_count.fetch_add(1, std::memory_order_release);
                    condition[i].notify_one();
                } else {
                    participating = true;
                }
            }
        }

        // If the main thread is participating, run the task
        if (participating) {
            task(rank_ct++, thread_dim, 0);
        }

        // Wait for all active threads to complete using condition variable
        if (active_thread_count.load(std::memory_order_acquire) > 0) {
            std::unique_lock<std::mutex> completion_lock(completion_mutex);
            completion_cv.wait(completion_lock, [this] {
                return active_thread_count.load(std::memory_order_acquire) == 0;
            });
        }
    }

    void set_perm_cpu(size_t cpu) {
        permanent_cpu = cpu;
    }

    template<class F>
    void operator()(bool mode, F&& f, __uint128_t mask = ~(__uint128_t)0) {
        if (mode) {

            if (mask == ~(__uint128_t)0) {
                mask = global_override_mask;
            }

            execute_parallel(std::forward<F>(f), global_override_mask);
        }
    }

    void set_override_mask(__uint128_t mask) {
        global_override_mask = mask;
    }

    __uint128_t get_override_mask() {
        return global_override_mask;
    }

    ~ThreadPool() {
        stop = true;
        
        for (size_t i = 1; i < workers.size() + 1; ++i) {
            condition[i].notify_one();
        }

        for (std::thread& worker : workers) {
            worker.join();
        }
    }
};

#endif