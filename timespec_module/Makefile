CC = g++ -std=c++14 -O0 -I.
FLAGS = -Wall -g -gdwarf-3
LIBS = -L. -lrt -lm -lclustering -fopenmp
CLUSTERING_OBJECTS = process_barrier.o generic_barrier.o timespec_functions.o
NIincludes = -I/usr/local/natinst/nidaqmxbase/include
NIlibs=-lnidaqmxbase
DIRS=build bin

#########################################################################
all: setup
	make -C build build

build: clustering_distribution

setup:
	$(shell mkdir -p $(DIRS))
	$(shell find . -name \*.cpp -exec cp {} build \;)
	$(shell find . -name \*.h -exec cp {} build \;)
	$(shell find . -name \*.hpp -exec cp {} build \;)
	$(shell find . -name Makefile -exec cp {} build \;)
#########################################################################

synthetic_task: synthetic_task.cpp
	$(CC) $(FLAGS) -fopenmp synthetic_task.cpp sharedMem.o task.o task_manager.o printBuffer.o thread_barrier.o schedule.o taskData.o -o synthetic_task $(LIBS)

synthetic_task_gd: synthetic_task_gd.cpp
	$(CC) $(FLAGS) -fopenmp synthetic_task_gd.cpp sharedMem.o task_manager.o printBuffer.o task.o thread_barrier.o  -o st_gd $(LIBS)

synthetic_task_gd_extra: synthetic_task_gd_extra.cpp
	$(CC) $(FLAGS) -fopenmp synthetic_task_gd_extra.cpp sharedMem.o task_manager.o printBuffer.o task.o thread_barrier.o  -o st_extra $(LIBS)

thread_barrier.o: thread_barrier.cpp
	$(CC) $(FLAGS) -c thread_barrier.cpp

clustering_distribution: libclustering.a sharedMem.o schedule.o scheduler.o task.o taskData.o task_manager.o thread_barrier.o printBuffer.o clustering_launcher synthetic_task james

libclustering.a: $(CLUSTERING_OBJECTS)
	ar rcsf libclustering.a $(CLUSTERING_OBJECTS)

task.o: task.cpp
	$(CC) $(FLAGS) -c task.cpp

task_manager.o: schedule.cpp schedule.cpp sharedMem.cpp task_manager.cpp
	$(CC) $(FLAGS) -fopenmp -c task_manager.cpp

process_barrier.o: process_barrier.cpp
	$(CC) $(FLAGS) -c process_barrier.cpp

timespec_functions.o: timespec_functions.cpp
	$(CC) $(FLAGS) -c timespec_functions.cpp

scheduler.o: scheduler.cpp
	$(CC) $(FLAGS) -c scheduler.cpp

taskData.o: taskData.cpp
	$(CC) $(FLAGS) -c taskData.cpp

schedule.o: schedule.cpp
	$(CC) $(FLAGS) -c schedule.cpp

sharedMem.o: sharedMem.cpp
	$(CC) $(FLAGS) -c sharedMem.cpp

generic_barrier.o: generic_barrier.cpp
	$(CC) $(FLAGS) -c generic_barrier.cpp

printBuffer.o: printBuffer.cpp
	$(CC) $(FLAGS) -c printBuffer.cpp

clustering_launcher: clustering_launcher.cpp
	$(CC) $(FLAGS) taskData.o schedule.o scheduler.o sharedMem.o clustering_launcher.cpp -o clustering_launcher $(LIBS)

james: james.cpp task_manager.o
	$(CC) $(FLAGS) james.cpp sharedMem.o scheduler.o schedule.o taskData.o task.o task_manager.o printBuffer.o thread_barrier.o -o james $(LIBS)
	cp james phil

clean:
	rm -f *.o  *.pyc libclustering.a clustering_launcher synthetic_task_gd synthetic_task synthetic_task_gd_extra mixed_crit_test james phil
	rm -rf ./build ./bin
