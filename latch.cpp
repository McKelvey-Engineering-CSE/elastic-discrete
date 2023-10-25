#include "latch.h"

void latch::init_latch(int value){

    std::lock_guard<std::mutex> lock(mut_);

    counter = value;
}

void latch::count_down() {

    std::lock_guard<std::mutex> lock(mut_);

    counter = counter - 1;
    cv_.notify_all();
}

bool latch::try_wait() {

    std::cerr << "COUNT: " << counter << std::endl;

    return !counter.load(std::memory_order_relaxed);
}

void latch::wait() {
        if (try_wait()) return;

        std::unique_lock<std::mutex> lock(mut_);
        cv_.wait(lock, [this](){ return try_wait(); });
    }

void latch::arrive_and_wait() {
    count_down();
    wait();
}