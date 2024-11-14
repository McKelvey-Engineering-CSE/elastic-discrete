##### Compiler Detection and Settings #################################################
NVCC := $(shell which nvcc 2> /dev/null)
NVCC := $(notdir $(NVCC))
HAS_NVCC := $(if $(filter nvcc,$(NVCC)),true,false)

# Common settings
COMMON_FLAGS := -std=c++20 -O0 -I. -g
COMMON_LIBS := -lrt -lm -L./libyaml-cpp/build/ -lyaml-cpp

# Include directories
HEADERS := $(addprefix -I ,$(shell find . -type d -not -path "*/\.*" | grep -v yaml))
HEADERS_WITH_YAML := $(addprefix -I ,$(shell find . -type d -not -path "*/\.*"))

# Architecture-specific flags
X86_64_ARCH := $(shell g++ -dumpmachine | grep x86_64)
ifneq (,$(X86_64_ARCH))
    ARCH_FLAGS := -mavx2
endif

# Compiler-specific settings
ifeq ($(HAS_NVCC),true)
    CC := nvcc
    FLAGS := $(COMMON_FLAGS) -Xcompiler -Wall -Xcompiler -gdwarf-3 $(HEADERS) -lcuda -lcudart
    LIBS := $(COMMON_LIBS) -Xcompiler -fopenmp
    ifneq (,$(X86_64_ARCH))
        FLAGS += -Xcompiler $(ARCH_FLAGS)
    endif
else
    CC := g++
    FLAGS := $(COMMON_FLAGS) $(HEADERS) -Wall -gdwarf-3
    LIBS := $(COMMON_LIBS) -fopenmp
    ifneq (,$(X86_64_ARCH))
        FLAGS += $(ARCH_FLAGS)
    endif
endif

##### Project Configuration ########################################################
TARGET_TASK := james
RTPS_FILE := ./target_task/james.yaml
CLUSTERING_OBJECTS := process_barrier.o generic_barrier.o timespec_functions.o process_primitives.o
BARRIER_OBJECTS := process_primitives.o generic_barrier.o process_barrier.o thread_barrier.o

##### Main Targets ##############################################################
.PHONY: all clean finish clean_libsmctrl libsmctrl

all: libsmctrl clustering_distribution finish regression_test_task

finish:
	mkdir -p ./bin
	cp $(TARGET_TASK) $(RTPS_FILE) ./clustering_launcher ./bin

clean: clean_libsmctrl
	rm -r ./bin *.o *.a $(TARGET_TASK) clustering_launcher synthetic_task || true

##### Conditional Targets and Rules #############################################
ifeq ($(HAS_NVCC),true)
clean_libsmctrl:
	cd ./libsmctrl && make clean && rm libsmctrl.o

libsmctrl:
	cd ./libsmctrl && make libsmctrl.a

taskData.o: taskData_real.o libsmctrl
	ld -relocatable taskData_real.o libsmctrl/libsmctrl.o -o taskData.o
else
clean_libsmctrl:
	@echo "Skipping libsmctrl clean (CUDA not available)"

libsmctrl:
	@echo "Skipping libsmctrl build (CUDA not available)"

taskData.o: taskData_real.o
	ld -relocatable taskData_real.o -o taskData.o
endif

##### Common Object Files ######################################################
timespec_functions.o: ./timespec_module/timespec_functions.cpp
	$(CC) $(FLAGS) -c $<

# Barrier module components
process_primitives.o: ./barrier_module/process_primitives.cpp
	$(CC) $(FLAGS) -c $<

generic_barrier.o: process_primitives.o ./barrier_module/generic_barrier.cpp
	$(CC) $(FLAGS) -c ./barrier_module/generic_barrier.cpp

process_barrier.o: ./barrier_module/process_barrier.cpp generic_barrier.o
	$(CC) $(FLAGS) -c $<

thread_barrier.o: ./barrier_module/thread_barrier.cpp generic_barrier.o process_barrier.o
	$(CC) $(FLAGS) -c $<

synthetic_task: ./task_module/synthetic_task.cpp shared_mem.o task.o task_manager.o print_library.o $(BARRIER_OBJECTS) schedule.o taskData.o timespec_functions.o
	$(CC) $(FLAGS) $^ -o $@ $(LIBS)

clustering: libclustering.a shared_mem.o schedule.o scheduler.o task.o taskData.o task_manager.o thread_barrier.o print_library.o clustering_launcher

clustering_distribution: clustering synthetic_task james

libclustering.a: $(CLUSTERING_OBJECTS)
	ar rcsf $@ $^

# Object compilation rules
task.o: ./task_module/task.cpp timespec_functions.o
	$(CC) $(FLAGS) -c $<

scheduler.o: ./scheduler_module/scheduler.cpp timespec_functions.o
	$(CC) $(FLAGS) -c $<

schedule.o: ./scheduler_module/schedule.cpp timespec_functions.o
	$(CC) $(FLAGS) -c $<

shared_mem.o: ./shared_memory_module/shared_mem.cpp
	$(CC) $(FLAGS) -c $<

taskData_real.o: ./task_module/taskData.cpp timespec_functions.o
	$(CC) $(FLAGS) -c $< -o $@

task_manager.o: ./main_binaries/task_manager.cpp timespec_functions.o process_barrier.o generic_barrier.o
	$(CC) $(FLAGS) $(LIBS) -c $<

print_library.o: print_module.o print_buffer.o
	ld -relocatable $^ -o $@

print_module.o: ./printing_module/print_module.cpp timespec_functions.o
	$(CC) $(FLAGS) -c $<

print_buffer.o: ./printing_module/print_buffer.cpp timespec_functions.o
	$(CC) $(FLAGS) -c $<

##### Final Targets ###########################################################
./libyaml-cpp/build/libyaml-cpp.a:
	cd libyaml-cpp && mkdir -p build && cd build && cmake .. && make

clustering_launcher: ./main_binaries/clustering_launcher.cpp ./libyaml-cpp/build/libyaml-cpp.a
	$(CC) $(FLAGS) $(HEADERS_WITH_YAML) timespec_functions.o taskData.o schedule.o scheduler.o shared_mem.o $(BARRIER_OBJECTS) ./main_binaries/clustering_launcher.cpp -o clustering_launcher $(LIBS)

james: ./target_task/james.cpp task_manager.o
	$(CC) $(FLAGS) ./target_task/james.cpp timespec_functions.o shared_mem.o scheduler.o schedule.o taskData.o task.o task_manager.o print_library.o $(BARRIER_OBJECTS) -o james $(LIBS)

regression_test_task: ./regression_test_task/regression_test_task.cpp clustering timespec_functions.o
	$(CC) $(FLAGS) ./regression_test_task/regression_test_task.cpp timespec_functions.o shared_mem.o scheduler.o schedule.o taskData.o task.o task_manager.o print_library.o $(BARRIER_OBJECTS) -o regression_test_task/regression_test_task $(LIBS)

##################################################################################
