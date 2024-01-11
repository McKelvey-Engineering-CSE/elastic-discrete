#ifndef ported_standard_library
#define ported_standard_library

/*************************************************************************

ported_standard_library.hpp

This file contains any C++17+ functions, structs, or objects that we 
reimplement for use in C++14. This allows for a reduced C++ std target
while still providing something to compile against if you want
certain C++20 functionality while designing tasks. This will be constantly
updated as we run into things we might want access to in various tasks.

namespace: std

class barrier:
      reimplementation of a standard C++20 barrier. Inherits from generic_barrier
      and requires a template type just like the standard version does

class latch:
      reimplementation of a standard C++20 latch. Inherits from generic_barrier
      and has the ability to reinitialize it disabled
        
**************************************************************************/
#include <cstddef>
#include "generic_barrier.h"

namespace std
{

/**************************************************************************
class latch:
      reimplementation of a standard C++20 latch. Inherits from generic_barrier
      and has the ability to reinitialize it disabled
**************************************************************************/
class latch : public generic_barrier {

private:

	using generic_barrier::init;

public:

    //standard latch constructor
    latch(size_t count_){
        count = count_;
        ret_function = nullptr;
        execute_function = false;

    }

    //return false only if counter = 0
    bool try_wait(){ return !count; }

    //decrement counter without blocking calling thread/process
    void count_down(){
        std::unique_lock<std::mutex> lock(mut);
        --count;
    }

    //wait at the barrier without decrementing
    void wait() {
        std::unique_lock<std::mutex> lock(mut);
        cv.wait(lock, [this] { return count == 0; });
    }

    //generic constantexpr function to return value
    static constexpr std::ptrdiff_t max(){ return (size_t)-1; }

};

/**************************************************************************
class latch:
      reimplementation of a standard C++20 latch. Inherits from generic_barrier
      and has the ability to reinitialize it disabled
**************************************************************************/
class barrier : public generic_barrier {

private:

    std::size_t expected;

    void init(){
        count = expected;
    }

    void return_function(){

        init();
        ret_function();

    }   

public:

    //standard latch constructor
    barrier(size_t count_, std::function<void()> returnFunction_){
        expected = count_;
        count = count_;
        ret_function = returnFunction_;
        scheduler_only = true;
        execute_function = true;
    }

    //not correctly implemented yet
    void arrive(){
        std::unique_lock<std::mutex> lock(mut);
        --count;
    }

    //decrement the expected number for all future iterations
    void arrive_and_drop(){
        std::unique_lock<std::mutex> lock(mut);
        --count;
        --expected;
    }

    //wait at the barrier without decrementing
    void wait() {
        std::unique_lock<std::mutex> lock(mut);
        cv.wait(lock, [this] { return count == 0; });
    }

    //generic constantexpr function to return value
    static constexpr std::ptrdiff_t max(){ return (size_t)-1; }

};
 




}

#endif