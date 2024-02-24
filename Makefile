##### Compiler Settings ##########################################################
CC = g++ -std=c++14 -O0 -I.
HEADERS = $(addprefix -iquote ,$(shell find . -type d -not -path "*/\.*"))
FLAGS = -Wall -g -gdwarf-3 $(HEADERS) -mavx2
LIBS = -L. -lrt -lm -lclustering -fopenmp
CLUSTERING_OBJECTS = process_barrier.o generic_barrier.o timespec_functions.o
##################################################################################

##### Task Configuration #########################################################
TARGET_TASK=james
RTPS_FILE=./target_task/james.rtps
##################################################################################

##### Rules ######################################################################
all: clustering_distribution finish

finish:
	mkdir ./bin
	cp $(TARGET_TASK) $(RTPS_FILE) ./clustering_launcher ./bin

clean:
	rm -r ./bin *.o *.a $(TARGET_TASK) clustering_launcher synthetic_task

synthetic_task: ./task_module/synthetic_task.cpp
	$(CC) $(FLAGS) -fopenmp ./task_module/synthetic_task.cpp shared_mem.o task.o task_manager.o print_library.o thread_barrier.o schedule.o taskData.o -o synthetic_task $(LIBS)

thread_barrier.o: ./barrier_module/thread_barrier.cpp
	$(CC) $(FLAGS) -c ./barrier_module/thread_barrier.cpp

clustering_distribution: libclustering.a shared_mem.o schedule.o scheduler.o task.o taskData.o task_manager.o thread_barrier.o print_library.o clustering_launcher synthetic_task james

libclustering.a: $(CLUSTERING_OBJECTS)
	ar rcsf libclustering.a $(CLUSTERING_OBJECTS)

task.o: ./task_module/task.cpp
	$(CC) $(FLAGS) -c ./task_module/task.cpp

task_manager.o: ./scheduler_module/schedule.cpp ./scheduler_module/schedule.cpp ./shared_memory_module/shared_mem.cpp ./main_binaries/task_manager.cpp
	$(CC) $(FLAGS) -fopenmp -c ./main_binaries/task_manager.cpp

process_barrier.o: ./barrier_module/process_barrier.cpp
	$(CC) $(FLAGS) -c ./barrier_module/process_barrier.cpp

timespec_functions.o: ./timespec_module/timespec_functions.cpp
	$(CC) $(FLAGS) -c ./timespec_module/timespec_functions.cpp

scheduler.o: ./scheduler_module/scheduler.cpp
	$(CC) $(FLAGS) -c ./scheduler_module/scheduler.cpp

taskData.o: ./task_module/taskData.cpp
	$(CC) $(FLAGS) -c ./task_module/taskData.cpp

schedule.o: ./scheduler_module/schedule.cpp
	$(CC) $(FLAGS) -c ./scheduler_module/schedule.cpp

shared_mem.o: ./shared_memory_module/shared_mem.cpp
	$(CC) $(FLAGS) -c ./shared_memory_module/shared_mem.cpp

generic_barrier.o: process_primitives.o ./barrier_module/generic_barrier.cpp
	$(CC) $(FLAGS) -c ./barrier_module/generic_barrier.cpp
	mv generic_barrier.o generic_barrier_inc.o
	ld -relocatable process_primitives.o generic_barrier_inc.o -o generic_barrier.o

process_primitives.o: ./barrier_module/process_primitives.cpp
	$(CC) $(FLAGS) -c ./barrier_module/process_primitives.cpp

print_library.o: print_module.o print_buffer.o
	ld -relocatable print_module.o print_buffer.o -o print_library.o

print_buffer.o: ./printing_module/print_buffer.cpp
	$(CC) $(FLAGS) -c ./printing_module/print_buffer.cpp

print_module.o: ./printing_module/print_module.cpp
	$(CC) $(FLAGS) -c ./printing_module/print_module.cpp 

clustering_launcher: ./main_binaries/clustering_launcher.cpp
	$(CC) $(FLAGS) taskData.o schedule.o scheduler.o shared_mem.o process_barrier.o ./main_binaries/clustering_launcher.cpp -o clustering_launcher $(LIBS)

james: ./target_task/james.cpp task_manager.o
	$(CC) $(FLAGS) ./target_task/james.cpp shared_mem.o scheduler.o schedule.o taskData.o task.o task_manager.o print_library.o thread_barrier.o -o james $(LIBS)
##################################################################################