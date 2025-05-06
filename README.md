# Elastic-Discrete Scheduler

Readme last updated: 6 May 2025 by Tyler Martin

## Description
This is a runtime system to run parallel elastic tasks with discrete candidate values of period T OR work C. 
Each task Tau has a constant span L, variable T or C, Elasticity coefficient E, and a finite set of discrete values of C or T

The scheduler implements the new heterogeneous discrete elastic framework, which is optimized for use with a CPU and NVIDIA GPU hybrid 
system. When calling the make file, if nvcc is installed, it will compile for use with the NVIDIA GPU. If nvcc is not found, it will compile
with g++. 

There are two versions of the core scheduler as well: one written in CUDA and one in standard C++. They are both contained within the same code, 
activated by macros at compile time. The scheduler is embarassingly parallel, and thus having a GPU in the system (even a terrible, 20$ one) is likely to
be faster than any CPU you pair with the system. 

Final note is that the scheduler can run in real or simulated modes. When in real mode, all tasks are unique processes and do some work as defined
by you. In simulated mode, the task file provided to the scheduler will be read in as usual, but the tasks associated with the parameters will never
be spawned. Instead the scheduler will pretend to be each task and do a fixed number of iterations, making reschedule requests just as they would. 
This exists so that you can test huge numbers of mode changes in one shot without having to wait for each task to actually do work. This is 
incredibly useful for debugging and extending the core scheduler behavior. To activate this mode, see below in the "Usage" section.

## Important Details
All classes implemented for thread/process management are all based on the implemented "generic_barrier" class, which is an process and thread safe recreation of the std::barrier/std::generic_barrier. This custom type enables us to keep to C++ 11/C++14 without issue as well as giving us fine-grained control over the abilities of the generic_barrier/barrier. Currently, the generic_barrier can take in a function poiner for the exiting processes or threads to execute when they leave the barrier. 

Priting is all controlled by a custom print function notated print_module::print(stream | buffer name | buffer name set, message/variables, ...)
(See section below for breakdown of printing module)

OpenMP can be used but is not strictly supported. There is a custom library in the omp_module which replaces the openMP library. Looking at the header file contained within should give enough insight to figure out how to use OMP-like semantics. If you must use OMP, it can be compiled with OMP support. However, thread pinning is up to you to do. You can access the cores granted to a task through the TaskData structures and pin threads yourself. 

As I continue working, this readme will be updated with more information

## Concurrency Primitives
All concurrency uses a custom p_mutex and p_condition_variable class. These classes mirror their std::mutex and std::condition_variable counterparts in functionality, but they ensure process safety without undefined behavior. They are really only used in the barriers, but as with all libraries, they can be used for synchronization within and between target tasks when compiling.

Classes Provided:
```
p_mutex

p_condition_variable
```

Functions Provided:
```
p_mutex::lock();

p_mutex::unlock();

p_condition_variable::wait(p_mutex);

p_condition_variable(p_mutex, bool (*Functionptr));

p_condition_variable::notify_all();
```

## Memory allocation
All memory allocation is handled through the shared_memory_module. This provides many differnt functions for easily creating, fetching, destroying, and detatching from shared memory. The signatures for each can be seen below:

Functions Provided:
```

//provides the basis for the key system. Can be used statically as well to generate keys from strings
template <typename T> key_t nameToKey(T name);

//allocate memory block (Args correspond to constructor values for the new block)
template <class T, typename... Args> T* allocate(std::string bufferName, Args&&... args);

//fetch memory block from key associated string
template <class T> T* fetch(std::string bufferName);

//detatch from memory associated with string
template <class T> int detatch(T* mem_seg);

//delete memory segment
template <class T> void delete_memory(std::string bufferName);
```

## Printing

Printing is handled by the "print_module" and can be accessed through the namespace of the same name. There are currently 3 unique "print" functions. they are all built off of std::cout/std::cerr, and therefore grants the ability to overload the "<<" operator in any classes we want. It however uses an ostream to ensure we get a printf-like thread-defined behavior with minimal performance penalty over std::cout/std::cerr. 

Classes Provided:
```
print_module::printBuffer

print_module::bufferSet(std::string buffer_name_one, std::string buffer_name_two, ...)
```

Functions Provided:
```
print(std::cout, message/variables, ...)

print(std::cerr, message/variables, ...)

print(const char[], message/variables, ...)

print(print_module::bufferSet, message/variables, ...)

print_module::printBuffer* createBuffer(std::string);

std::vector<print_module::printBuffer*> createBuffer(bufferSet);

int deleteBuffer(std::string);

int deleteBuffer(print_module::bufferSet);
```

The main difference with normal printing is that you can print to either an std::ostream-compliant interface (std::cout\std::cerr), a single mmap allocated print_module::printBuffer or a set of printBuffers notated by a print_module::bufferSet which is a unique data structure containing the names of buffers that you want to control as a single unit.

This printing method can be imported into any object in the system and used. It ensures a thread-safe and thread-ready method for printing which 
guarantees no messages are interleaved when printing to any std::ostream interface as well as enabling a printing method for inter-process 
communication for future endeavors

NOT FINISHED

## Usage

Ensure you have a task/program with 3 fields and a task structure: 

```
int init(int argc, char *argv[]);

int run(int argc, char *argv[]);

int finalize(int argc, char *argv[]);

task_t task = { init, run, finalize };
```

Once you have this structure, compile your task against the task_manager object to enable the scheduler to use it.

Next define a configuration YAML file. File structure:
```
schedulable: true/false
maxRuntime: {sec: 0, ns: 0} # optional
tasks:
  - program:
      name: "executable file name"
      args: "these are arguments"
    elasticity: 150 # For non-elastic mode set to 1
    maxIterations: 100 # Optional
    priority: 1 # Optional
    modes:
      - work: {sec: 5, nsec: 0}
        span: {sec: 1, nsec: 0}
        period: {sec: 0, nsec: 3000000}
```

A task will run until either it reaches the maximum number of iterations, or the global maximum runtime is reached - if neither is specified it will run forever.

To emulate the behavior of the old clustering launcher, set `elasticity: 1` for every task.

`priority` controls the priority given to the kernel (under the SCHED_RR scheduler). If no priority is set, `7` is used as the default. Note that during task sleep and finalization, the set priority is ignored.

Finally, to run it, navigate to the "bin" directory and call

```
./clustering_launcher ./<input_file_name>.yaml
```

and to run it in simulated mode

```
./clustering_launcher ./<input_file_name>.yaml SIM
```


## Elastic Discrete Legacy Description (For Use With Legacy Cybermech)
```
Author: James Orr
Date: 5 September, 2018

This is a runtime system to run parallel elastic tasks with discrete candidate values of period T OR work C. 
Each task Tau has a constant span L, variable T or C, Elasticity coefficient E, and a finite set of discrete values of C or T

In order to get this running on Cybermech run the following 2 commands each session (also in init.sh): 
export PATH=/home/james/bin:$PATH &&  export LD_LIBRARY_PATH=/home/james/lib64:$LD_LIBRARY_PATH && export GOMP_SPINCOUNT=0
```
