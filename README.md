# Elastic-Discrete Scheduler

Readme last updated: 24 February 2024 by Tyler Martin

## Description
This is a runtime system to run parallel elastic tasks with discrete candidate values of period T OR work C. 
Each task Tau has a constant span L, variable T or C, Elasticity coefficient E, and a finite set of discrete values of C or T

## Important Details
All classes implemented for thread/process management are all based on the implemented "generic_barrier" class, which is an process and thread safe recreation of the std::barrier/std::generic_barrier. This custom type enables us to keep to C++ 11/C++14 without issue as well as giving us fine-grained control over the abilities of the generic_barrier/barrier. Currently, the generic_barrier can take in a function poiner for the exiting processes or threads to execute when they leave the barrier. 

All barriers currently spin on the value associated with processes that have entered the barrier. This will be exchanged for a pthread condition variable once a better way to integrate it has been designed.

Priting is all controlled by a custom print function notated print_module::print(stream | buffer name | buffer name set, message/variables, ...)
(See section below for breakdown of printing module)

All code is currently being rewritten to ensure only C++11/C++14 idioms are used, and that we do not have any issues with OpenMP and the more modern C++ versions (there have been no issues so far).

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

## Testing

### Adding a testcase

First, set up the synthetic tasks you wish to test. A `regression_test_task` has been provided for this purpose; it takes arguments of the form `<nsec> <mode_count> <mode_change_interval>`. On every iteration, it will spin for `nsec` nanoseconds; after every `<mode_change_interval>`, it will advance to the next mode. On every iteration, it prints out the number of cores it has been assigned; if you write your own synthetic task, you should print the core assignments in the same format.

Then, create a `.yaml` file describing your task system and save it in `regression_tests/tests/`. Make sure to use the correct relative paths to refer to your compiled synthetic tasks. Compile both `clustering_launcher` and your synthetic tasks. Run your testcase and save the expected output in the same folder, with a `.txt` file extension, like so:

```
cd regression_tests/build/
../../clustering_launcher ../tests/basic.yaml &> ../tests/basic.txt
```

Finally, add a testcase to `test_runner.cpp` describing the properties you wish to check. For example, to check that all the core assignments match the expected results:

```
TEST(Scheduler, BasicAssignment) {
    clusteringRunResult result = runClustering("basic");
    ASSERT_EQ(result.exit_code, 0);
    assert_same_core_assignments(parse_core_assignments(result.stdout), parse_core_assignments(get_expected_output("basic")));
}
```

### Executing tests

```
cd regression_tests
mkdir build
cd build
cmake ../
make
./test_runner ../../clustering_launcher ../tests/
```

The first argument to `test_runner` can be replaced with the path to any instance of `clustering_launcher` that you wish to test. You can also add additional arguments; they will be passed through to [gtest](https://google.github.io/googletest/).

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
