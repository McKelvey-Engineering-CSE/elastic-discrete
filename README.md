# RT-HEART (Real-Time Heterogeneous Elastic Adaptive Runtime)

[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-green.svg)](https://developer.nvidia.com/cuda-zone)

**Last updated:** 1 August 2025 by Tyler Martin

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Heterogeneous CPU Support](#heterogeneous-cpu-support)
- [Usage](#usage)
- [API Reference](#api-reference)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Testing](#testing)

## Overview

RT-HEART is a high-performance runtime system for executing parallel elastic tasks with discrete candidate values of period T or work C. Each task Tau has a constant span L, variable T or C, elasticity coefficient E, and a finite set of discrete values of C or T where each set represents one unique mode the task can be operating in.

The scheduler implements a novel heterogeneous discrete elastic framework that supports **four distinct accelerator types**:
- **Processor A**: Primary accelerator (typically CPU cores managed by the OMP replacement library)
- **Processor B**: Secondary accelerator (e.g., GPU, FPGA, specialized cores)
- **Processor C**: Tertiary accelerator (e.g., additional GPU, neural engine)
- **Processor D**: Quaternary accelerator (e.g., custom ASIC, additional FPGA)

### Key Capabilities
- **Multi-Accelerator Support**: Arbitrary combination of CPU, GPU, FPGA, and custom accelerators
- **Heterogeneous CPU Support**: Specialized handling for CPUs with performance-tiered cores
- **GPU Acceleration**: GPU accelerated calculation of system configuration on mode change
- **Elasticity Based Scheduling**: Mode objective function follows the mature elasticity framework
- **Real/Simulated Modes**: Production execution or fast simulation for testing

## Features

RT-HEART's core innovation lies in its ability to handle dynamic resource reallocation across arbitrary heterogeneous systems. When a task requests a mode change, the scheduler doesn't simply pause execution and reassign resources—instead, it calculates an optimal sequence of resource handoffs that allows the requesting task to reach its new mode while maximizing overall system efficiency.

The system achieves this through a specialized knapsack scheduler that can be GPU-accelerated for millisecond response times. This scheduler analyzes the current system state, evaluates all possible resource allocation combinations, and generates a resource handoff graph that ensures no task needs to stop execution during the transition. The elasticity framework provides the theoretical foundation, allowing tasks to operate with discrete candidate values for period T or work C while maintaining real-time guarantees.

Key capabilities include:
- **Millisecond system state calculation** through GPU-accelerated optimization
- **Deadlock-free resource management** with optimal transfer sequences
- **Automatic heterogeneous CPU optimization** for performance-tiered cores
- **Real-time mode transitions** without execution interruption

## Quick Start

### 1. Create a Task
```cpp
// my_task.cpp
#include "task.h"

int init(int argc, char *argv[]) {
    // Initialize your task
    return 0;
}

int run(int argc, char *argv[]) {
    // Main task execution
    return 0;
}

int finalize(int argc, char *argv[]) {
    // Cleanup
    return 0;
}

void update_core_B(__uint128_t mask) {
    // Update accelerator B assignment
}

void update_core_C(__uint128_t mask) {
    // Update accelerator C assignment
}

void update_core_D(__uint128_t mask) {
    // Update accelerator D assignment
}

task_t task = { init, run, finalize, update_core_B, update_core_C, update_core_D };
```

**Function Descriptions:**
- **`init()`**: Runs once when the system starts before each task begins working
- **`run()`**: Main body of the task which can have different modes depending on how you define it
- **`finalize()`**: Executed once only when the task's execution time completes
- **`update_core_B/C/D()`**: Can use the corresponding masks to make use of the cores that task has been assigned while allowing the user to define how it uses them

### 2. Define Configuration
```yaml
# my_config.yaml
schedulable: true
explicit_sync: false
maxRuntime: {sec: 30, nsec: 0}
processorConfiguration: {A: 1A, B: 0.75A, C: 1C, D: 1D}
tasks:
  - program:
      name: "my_task"
      args: "arg1 arg2"
    elasticity: 2
    priority: 7
    modes:
      - work_A: {sec: 0, nsec: 100000000}
        span_A: {sec: 0, nsec: 50000000}
        period: {sec: 0, nsec: 200000000}
```

**Mode Parameters:**
For each mode, you can specify `work_(A-D)` and `span_(A-D)` parameters that determine:
- **Resource Allocation**: How the mode is allocated resources across accelerators
- **Deadline Requirements**: The deadline it must meet
- **Resource Release**: How quickly it releases resources (the period)

### 3. Run the Scheduler
```bash
# Real mode
./bin/clustering_launcher ./my_config.yaml

# Simulation mode
./bin/clustering_launcher ./my_config.yaml SIM
```

## Heterogeneous CPU Support

RT-HEART provides specialized support for heterogeneous CPUs with performance-tiered cores (e.g., Intel's P-cores and E-cores, ARM big.LITTLE architecture). This is achieved through the `processorConfiguration` field in the YAML configuration.

### Configuration Format
```yaml
processorConfiguration:
  A: 1A      # Processor A is independent (baseline)
  B: 0.75A   # Processor B is 75% as fast as processor A
  C: 1C      # Processor C is independent
  D: 1D      # Processor D is independent
```

### How It Works
When you specify that a processor is equivalent to processor A with a performance ratio, the scheduler:

1. **Analyzes the DAG**: Examines the canonical form of your task's Directed Acyclic Graph
2. **Calculates Completion Time**: Determines time-to-completion across variable-speed processors
3. **Optimizes Allocation**: Generates safe core allocations for each processor type
4. **Accounts for Heterogeneity**: Considers performance differences when scheduling

### Example Use Case
For a system with:
- **Processor A**: Performance cores (P-cores) - baseline performance
- **Processor B**: Efficiency cores (E-cores) - 75% of P-core performance
- **Processor C**: GPU accelerator - independent performance
- **Processor D**: FPGA accelerator - independent performance

The scheduler will automatically account for the performance difference between P-cores and E-cores when calculating optimal task assignments and mode transitions.

### Benefits
- **Automatic Optimization**: No manual core assignment needed
- **Performance-Aware Scheduling**: Tasks are assigned based on actual performance ratios
- **Elastic Adaptation**: System can adapt to changing performance characteristics
- **Deadline Compliance**: Ensures real-time guarantees across heterogeneous cores

## Usage

### Task Structure
Every task must implement six functions:
```cpp
int init(int argc, char *argv[]);    // Initialization
int run(int argc, char *argv[]);     // Main execution loop
int finalize(int argc, char *argv[]); // Cleanup
void update_core_B(__uint128_t mask); // Update accelerator B assignment
void update_core_C(__uint128_t mask); // Update accelerator C assignment
void update_core_D(__uint128_t mask); // Update accelerator D assignment
```

### YAML Configuration Format
```yaml
schedulable: true/false                                     # Whether tasks can be scheduled
explicit_sync: false                                        # Optional explicit synchronization
maxRuntime: {sec: 0, ns: 0}                                 # Optional global runtime limit (Needed if no iteration count)
processorConfiguration:  {A: 1A, B: 0.75A, C: 1C, D: 1D}    # Processor description (see above)

tasks:
  - program:
      name: "executable_name"        # Task binary name
      args: "command line arguments" # Task arguments
    elasticity: 150                  # Elasticity coefficient (1 for non-elastic)
    maxIterations: 100               # Optional iteration limit
    priority: 1                      # Optional SCHED_RR priority (default: 7)
    modes:                           # Available execution modes (candidate modes)
      - work_A: {sec: 5, nsec: 0}    # Work for accelerator A
        span_A: {sec: 1, nsec: 0}    # Span for accelerator A
        work_B: {sec: 3, nsec: 0}    # Work for accelerator B (optional)
        span_B: {sec: 1, nsec: 0}    # Span for accelerator B (optional)
        work_C: {sec: 2, nsec: 0}    # Work for accelerator C (optional)
        span_C: {sec: 1, nsec: 0}    # Span for accelerator C (optional)
        work_D: {sec: 1, nsec: 0}    # Work for accelerator D (optional)
        span_D: {sec: 1, nsec: 0}    # Span for accelerator D (optional)
        period: {sec: 5, nsec: 0}    # Task period
```

### Execution Modes
- **Real Mode**: Tasks run as actual processes with real work
- **Simulation Mode**: Tasks are simulated for fast testing and debugging

## API Reference

### Additional Task Functions
```cpp
// Request mode change and system reschedule
void modify_self(int new_mode);

// Get current execution mode
int get_current_mode();

// Set cooperative behavior
void set_cooperative(bool cooperative);
```

**Function Descriptions:**
- **`modify_self(int new_mode)`**: Requests a mode change and triggers a reallocation of system resources to enable the mode change for the requesting task
- **`get_current_mode()`**: Returns the mode the system has acknowledged the task is in (this is important since what mode the system thinks the task is in is all that really matters - if the system doesn't know a task is in a specific mode, it won't allocate resources for that mode)
- **`set_cooperative(bool cooperative)`**: Marks whether the task is willing to have its mode lowered to a lower utilization mode to free up resources for another task which is requesting resources for its own mode change

### Task Variables
```cpp
// Current accelerator assignments
__uint128_t processor_A_mask;  // Accelerator A resource mask
__uint128_t processor_B_mask;  // Accelerator B resource mask
__uint128_t processor_C_mask;  // Accelerator C resource mask
__uint128_t processor_D_mask;  // Accelerator D resource mask
```

**Example Usage**: The masks are always current after mode changes and can be safely referenced in update functions. For example, in `james.cpp`:

```cpp
void update_core_B(__uint128_t mask) {
    // For a hybrid CPU, omp_replacement controls both A and B cores
    // Update the mask to reflect the B cores the task owns as well
    omp.set_override_mask(processor_A_mask | (processor_B_mask >> NUM_PROCESSOR_A));
}
```


## System Requirements

### Minimum Requirements
- **OS**: Linux (tested on Fedora 41, Ubuntu 20.04+)
- **Compiler**: GCC 9+ or Clang 10+ with C++20 support
- **CMake**: 3.16+ for building yaml-cpp dependency
- **Memory**: 4GB RAM minimum, 8GB+ recommended

### Optional Requirements
- **CUDA Toolkit**: 12.0+ for GPU acceleration

### Dependencies
- **yaml-cpp**: YAML parsing library (included as submodule)
- **POSIX RT**: Real-time scheduling capabilities
- **CUDA Runtime**: GPU acceleration (optional)

## Installation

### Prerequisites
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential cmake

# For GPU support (optional)
sudo apt-get install nvidia-cuda-toolkit
```

### Building from Source
```bash
# Clone the repository
git clone https://github.com/McKelvey-Engineering-CSE/elastic-discrete
cd elastic-discrete

# Initialize and update submodules
git submodule update --init --recursive

# Build RT-HEART
make all
```

### Build Options
The Makefile automatically detects your environment:
- **CUDA Available**: Compiles with GPU acceleration
- **CUDA Not Available**: Falls back to CPU-only compilation

## Testing

### Running Tests
```bash
cd testing_module/bin/
./clustering_launcher ./james.yaml
```

Starts a test configuration with a heterogeneous system comprised of 16 A cores, 16 B cores and 1 accelerator C. Spawns tasks which pretend to work via spinning with their resources and request mode changes as they execute.


**Note**: This is a research-grade system designed for heterogeneous real-time computing. For production use, additional testing and validation may be required.