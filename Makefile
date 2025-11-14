##### Compiler Detection and Settings #################################################
NVCC := $(shell which nvcc 2> /dev/null)
NVCC := $(notdir $(NVCC))
HAS_NVCC := $(if $(filter nvcc,$(NVCC)),true,false)

# Common settings
COMMON_FLAGS := -std=c++20 -O0 -I. -g
COMMON_LIBS := -lrt -lm -L./libyaml-cpp/build/ -lyaml-cpp -L /usr/lib64/

# Omp library control
OMP_LIB := -DOMP_OVERRIDE #-DPRETTY_PRINTING

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
    CC := nvcc $(OMP_LIB)
    FLAGS := $(COMMON_FLAGS) -arch=native --expt-relaxed-constexpr -Xcompiler -Wall -Xcompiler -gdwarf-3 $(HEADERS) -lcuda -lcudart -Xcompiler -mcmodel=medium -Xcompiler "-mavx2 -march=native -mfma -lopenblaso" -lopenblaso
    LIBS := $(COMMON_LIBS) -Xcompiler -fopenmp -L./omp_module -Xlinker -rpath,./omp_module -lcublas
	NVCC_OVERRIDE := --x=cu
    ifneq (,$(X86_64_ARCH))
        FLAGS += -Xcompiler $(ARCH_FLAGS)
    endif
else
    CC := g++ $(OMP_LIB)
    FLAGS := $(COMMON_FLAGS) $(HEADERS) -Wall -gdwarf-3
    LIBS := $(COMMON_LIBS) -fopenmp -L./omp_module -Wl,-rpath,./omp_module
    ifneq (,$(X86_64_ARCH))
        FLAGS += $(ARCH_FLAGS)
    endif
endif

##### Project Configuration ########################################################
TARGET_TASK := james
RTPS_FILE := ./example_task/james.yaml
CLUSTERING_OBJECTS := process_barrier.o generic_barrier.o timespec_functions.o process_primitives.o
BARRIER_OBJECTS := process_primitives.o generic_barrier.o process_barrier.o thread_barrier.o

##### Main Targets ##############################################################
.PHONY: all clean finish

all: clustering_distribution finish

finish: clustering_distribution james
	mkdir -p ./testing_module/bin
	cp $(TARGET_TASK) $(RTPS_FILE) ./clustering_launcher ./yaml_parser ./testing_module/bin

clean:
	rm -r ./testing_module/bin *.o *.a $(TARGET_TASK) clustering_launcher yaml_parser synthetic_task || true

taskData.o: taskData_real.o 
	ld -relocatable taskData_real.o -o taskData.o

timespec_functions.o: ./timespec_module/timespec_functions.cpp
	$(CC) $(NVCC_OVERRIDE) $(FLAGS) -c $<

# Barrier module components
process_primitives.o: ./barrier_module/process_primitives.cpp
	$(CC) $(NVCC_OVERRIDE) $(FLAGS) -c $<

generic_barrier.o: process_primitives.o ./barrier_module/generic_barrier.cpp
	$(CC) $(FLAGS) -c ./barrier_module/generic_barrier.cpp

process_barrier.o: ./barrier_module/process_barrier.cpp generic_barrier.o
	$(CC) $(NVCC_OVERRIDE) $(FLAGS) -c $<

thread_barrier.o: ./barrier_module/thread_barrier.cpp generic_barrier.o process_barrier.o
	$(CC) $(NVCC_OVERRIDE) $(FLAGS) -c $<

clustering_distribution: libclustering.a schedule.o scheduler.o task.o taskData.o task_manager.o thread_barrier.o print_library.o clustering_launcher james

libclustering.a: $(CLUSTERING_OBJECTS)
	ar rcsf $@ $^

# Object compilation rules
task.o: ./task_module/task.cpp timespec_functions.o
	$(CC) $(NVCC_OVERRIDE) $(FLAGS) -c $< $(LIBS)

scheduler_piece.o: ./scheduler_module/scheduler.cpp timespec_functions.o
	$(CC) $(NVCC_OVERRIDE) $(FLAGS) -c $< -o $@

#force link it with libsmctrl.o
scheduler.o: scheduler_piece.o libsmctrl/libsmctrl.o
	ld -relocatable $^ -o $@

schedule.o: ./scheduler_module/schedule.cpp timespec_functions.o
	$(CC) $(NVCC_OVERRIDE) $(FLAGS) -c $<

taskData_real.o: ./task_module/taskData.cpp timespec_functions.o
	$(CC) $(NVCC_OVERRIDE) $(FLAGS) -c $< -o $@

task_manager.o: ./main_binaries/task_manager.cpp timespec_functions.o process_barrier.o generic_barrier.o
	$(CC) $(NVCC_OVERRIDE) $(FLAGS) $(LIBS) -c $<

print_library.o: print_module.o print_buffer.o
	ld -relocatable $^ -o $@

print_module.o: ./printing_module/print_module.cpp timespec_functions.o
	$(CC) $(NVCC_OVERRIDE) $(FLAGS) -c $<

print_buffer.o: ./printing_module/print_buffer.cpp timespec_functions.o
	$(CC) $(NVCC_OVERRIDE) $(FLAGS) -c $<

##### Final Targets ###########################################################
./libyaml-cpp/build/libyaml-cpp.a:
	cd libyaml-cpp && mkdir -p build && cd build && cmake .. && make

yaml_parser: ./main_binaries/yaml_parser.cpp ./libyaml-cpp/build/libyaml-cpp.a timespec_functions.o taskData.o schedule.o scheduler.o $(BARRIER_OBJECTS)
	$(CC) $(FLAGS) $(HEADERS_WITH_YAML) timespec_functions.o taskData.o schedule.o scheduler.o $(BARRIER_OBJECTS) ./main_binaries/yaml_parser.cpp -o yaml_parser $(LIBS)

clustering_launcher-bin: ./main_binaries/clustering_launcher.cpp
	$(CC) $(FLAGS) $(NVCC_OVERRIDE) ./main_binaries/clustering_launcher.cpp -c $< $(LIBS)

clustering_launcher: yaml_parser clustering_launcher-bin timespec_functions.o taskData.o schedule.o scheduler.o $(BARRIER_OBJECTS)
	$(CC) $(FLAGS)  timespec_functions.o taskData.o schedule.o scheduler.o $(BARRIER_OBJECTS) clustering_launcher.o -o clustering_launcher $(LIBS)

james-bin: ./example_task/james.cpp 
	$(CC) $(FLAGS) $(NVCC_OVERRIDE) ./example_task/james.cpp -c $< $(LIBS)

james: james-bin task_manager.o print_library.o $(BARRIER_OBJECTS) timespec_functions.o schedule.o taskData.o task.o
	$(CC) $(FLAGS) james.o timespec_functions.o schedule.o taskData.o task.o libsmctrl/libsmctrl.o task_manager.o print_library.o $(BARRIER_OBJECTS) -o james $(LIBS)

