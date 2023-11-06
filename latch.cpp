#include "latch.h"

void latch::init_latch(int in){
    std::lock_guard<std::mutex> lock(mut);
    count = in;
}

void latch::arrive_and_wait()
{
    //lock mutex
    std::unique_lock<std::mutex> lock(mut);

    //check if exit or wait
    if (--count == 0) {
        cv.notify_all();
    } 
    
    else {
        cv.wait(lock, [this] { return count == 0;});
    }
}