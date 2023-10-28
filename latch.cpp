#include "latch.h"

void latch::init_latch(int value){

    print(std::cerr, "Latch being reinit to value:", value, "\n");

    std::lock_guard<std::mutex> lock(mut_);

    counter.store(value, std::memory_order_seq_cst);
}

void latch::count_down() {

    std::lock_guard<std::mutex> lock(mut_);

    counter.store(counter.load(std::memory_order_seq_cst) - 1, std::memory_order_seq_cst);

    print(std::cerr, "BARRIER JUST DECREMENTED, value: ", counter.load(std::memory_order_seq_cst), "\n");

    cv_.notify_all();
}

bool latch::try_wait() {
    return (counter.load(std::memory_order_seq_cst) != 0);
}

void latch::wait() {

    print(std::cerr, "Entering wait\n");
    if (!try_wait()) return;
    print(std::cerr, "couldn't exit, waiting now\n");

    while(try_wait());
    //std::unique_lock<std::mutex> lock(mut_);
    //cv_.wait(lock, [=](){print(std::cerr, "tried to unlock\n"); return (!try_wait());});
    print(std::cerr, "I'm Free!!\n");

}

void latch::arrive_and_wait() {

    count_down();
    wait();

    print(std::cerr, "FREED MYSELF\n");
}