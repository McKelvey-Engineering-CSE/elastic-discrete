# Elastic-Discrete Scheduler

Readme last updated: 11 January 2024 by Tyler Martin

## Description
This is a runtime system to run parallel elastic tasks with discrete candidate values of period T OR work C. 
Each task Tau has a constant span L, variable T or C, Elasticity coefficient E, and a finite set of discrete values of C or T

## Important Details
All classes implemented for thread/process management are all based on the implemented "generic_barrier" class, which is an process and thread safe recreation of the std::barrier/std::generic_barrier. This custom type enables us to keep to C++ 11/C++14 without issue as well as giving us fine-grained control over the abilities of the generic_barrier/barrier. Currently, the generic_barrier can take in a function poiner for the exiting processes or threads to execute when they leave the barrier. 

Priting is all controlled by a custom print function notated print_module::print(stream | buffer name | buffer name set, message/variables, ...)
(See section below for breakdown of printing module)

All code is currently being rewritten to ensure only C++11/C++14 idioms are used, and that we do not have any issues with OpenMP and the more modern C++ versions (there have been no issues so far).

As I continue working, this readme will be updated with more information

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

Once you have this set up, generate a scheduling file (.rtps) file that is the same name as the executable
(The Cybermech repository grants you a way to turn a .rtp file into a .rtps file for elastic scheduling)

RTPS File structure
```
<Taskset schedulable?>
<number of tasks> <s to run> <ns to run>
<process name> <process args (may be null)> 
<elastic coefficient> <number of modes of operation (1+)> [<work seconds> <work nanoseconds> <span seconds> <span nanoseconds> <period seconds> <period nanoseconds> (Repeat once per mode of operation)]
##REPEAT LINES 3-4 as needed. Together they form a task.
```

Once you have a .rtps file created, you can schedule and run your task using this scheduler. 

## Elastic Discrete Legacy Description
```
Author: James Orr
Date: 5 September, 2018

This is a runtime system to run parallel elastic tasks with discrete candidate values of period T OR work C. 
Each task Tau has a constant span L, variable T or C, Elasticity coefficient E, and a finite set of discrete values of C or T

See example.rtps for taskset representation.

In order to get this running on Cybermech run the following 2 commands each session (also in init.sh): 
export PATH=/home/james/bin:$PATH &&  export LD_LIBRARY_PATH=/home/james/lib64:$LD_LIBRARY_PATH && export GOMP_SPINCOUNT=0
```