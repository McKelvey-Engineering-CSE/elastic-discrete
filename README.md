# RT-HEART

Readme last updated: 31 July 2025 by Tyler Martin

## Description
This is a runtime system to run parallel elastic tasks with discrete candidate values of period T OR work C. 
Each task Tau has a constant span L, variable T or C, Elasticity coefficient E, and a finite set of discrete values of C or T

The scheduler implements the new heterogeneous discrete elastic framework, which supports **four distinct processor types**:
- **Processor A**: Primary CPU cores managed by the OMP replacement library
- **Processor B**: Additional processor 
- **Processor C**: Additional processor
- **Processor D**: Additional processor

Heterogeneous CPUs: CPUs which have the A-B-C core tiering system are fully supported. For more information, see below.

The system is optimized for use with heterogeneous computing environments including CPUs and NVIDIA GPUs. When calling the make file, if nvcc is installed, it will compile for use with the NVIDIA GPU. If nvcc is not found, it will compile with g++. 

There are two versions of the core scheduler as well: one written in CUDA and one in standard C++. They are both contained within the same code, 
activated by macros at compile time. The scheduler is embarassingly parallel, and thus having a GPU in the system (even a terrible, 20$ one) is likely to be faster than any CPU you pair with the system. 

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

**Processor Management**: Processor A is automatically managed by the OMP replacement library, while processors B, C, and D require manual management through exposed task functions.

As I continue working, this readme will be updated with more information

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
explicit_sync: false # Optional
maxRuntime: {sec: 0, ns: 0} # optional
processorConfiguration: {A: 1A, B: 0.75A, C: 1C, D: 1D} # Defines processor relationships
tasks:
  - program:
      name: "executable file name"
      args: "these are arguments"
    elasticity: 150 # For non-elastic mode set to 1
    maxIterations: 100 # Optional
    priority: 1 # Optional
    modes:
      - work_A: {sec: 5, nsec: 0}    # Work for processor A
        span_A: {sec: 1, nsec: 0}    # Span for processor A
        work_B: {sec: 3, nsec: 0}    # Work for processor B (optional)
        span_B: {sec: 1, nsec: 0}    # Span for processor B (optional)
        work_C: {sec: 2, nsec: 0}    # Work for processor C (optional)
        span_C: {sec: 1, nsec: 0}    # Span for processor C (optional)
        work_D: {sec: 1, nsec: 0}    # Work for processor D (optional)
        span_D: {sec: 1, nsec: 0}    # Span for processor D (optional)
        period: {sec: 0, nsec: 3000000}
```

**Hybrid Processor Configuration**: The `processorConfiguration` field defines the relationship between processor types:
- `A: 1A` - Processor A is independent
- `B: 0.75A` - Processor B has 75% of the performance of processor A
- `C: 1C` - Processor C is independent  
- `D: 1D` - Processor D is independent

If a processor is heterogeneous and contains different cores within it, often treating these cores as equivalent is not useful. In the yaml file, if you specify that core B is 75% as fast as core A, the scheduler will calculate mode resource usages considering the canonical DAG for your task and generate safe core allocations for each core type present in the CPU. In the example above, we have a system with an A-B processor like a core Ultra 9 285k and two accelerators, maybe an Nvidia GPU and an Xilinx FPGA.

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

## Functions You Can Access From A Task

Each task has a few different functions it can call do affect or check various things in the system.
These functions are as follows:

```
void modify_self(int);
```
Allows a task to modify the mode it is currently running in and request a full system reschedule. 
Pass in the number of the mode you would like to transition to and the scheduler will immediately 
calculate and execute a new system schedule. If there is no way to get the task to that mode, 
the task will be rescheduled in its current mode.

```
int get_current_mode();
```
Call to check what mode the task is currently executing in.

```
void set_cooperative(bool);
```
Allows setting of whether or not the task is cooperative. If a task is cooperative, then the task
will allow itself to be changed to a different mode upon system reschedules to help the rest of the system
achieve the new running state. If set to false. then the system will not force this task to change modes unless
the task requests a mode change.

```
void allow_change();
```
To be used after set_coopertative is run if you desire the system to calculate a new schedule with the task
in question now being allowed to cooperate and have its mode changed. 

**Processor Management Functions**:
```
void update_core_B(__uint128_t mask);
void update_core_C(__uint128_t mask);
void update_core_D(__uint128_t mask);
```
These functions allow tasks to manually update their processor assignments for processors B, C, and D.
Processor A is automatically managed by the OMP replacement library and does not require manual updates.
The mask parameter specifies which cores are assigned to the task (least significant bit is core 0).

## Useful Variables You Can Access From A Task

Each task also has different variables that they own which reflect different parts of their current 
execution scheme and operating mode. The processor mask variables are:

```
__uint128_t processor_A_mask;
__uint128_t processor_B_mask;
__uint128_t processor_C_mask;
__uint128_t processor_D_mask;
```

These can be called from anywhere and will always contain the current processor core masks for each processor type.
The masks detail which processor cores this task currently has owned by it. Least significant bit is core 0 up to core
128 in the most significant bit.

## Examples

Within the "example_task" directory you can find an example task "james.cpp" as well as an example "james.yaml" configuration file.

If you're wondering why they are called "james" still: I can't bring myself to rename them to something like "example", just too much fun
having the "james" test case.

NOTE: The yaml file and process do not need to have the same name. They just do in this example case.

If you just build the scheduler as is provided, you will get a new directory created called "bin" which contains a copy of all the binaries needed
to run the simple "james" example and see the scheduler working.

# Additional Provided Libraries


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
template <typename Arg, typename... Args>
void buffered_print(std::ostringstream& out, Arg&& arg, Args&&... args);

template <typename Arg, typename... Args>
void print(std::ostream& out, Arg&& arg, Args&&... args);

template <typename Arg, typename... Args>
void task_print(std::ostream& out, Arg&& arg, Args&&... args);

template <typename... Args>
void flush(std::ostream& out, std::ostringstream& buff, Args&&... args);

template <typename Arg, typename... Args>
void print(const char bufferChar[], Arg&& arg, Args&&... args);

template <typename Arg, typename... Args>
void print(buffer_set bufferNames, Arg&& arg, Args&&... args);
```

This printing method can be imported into any object in the system and used. It ensures a process-safe and thread-ready method for printing which 
guarantees no messages are interleaved when printing to any std::ostream interface as well as enabling a printing method for inter-process 
communication for future endeavors