CC = g++ -std=c++20 -O0
FLAGS = -Wall -g -gdwarf-3
LIBS = -L. -lrt -lm -lclustering -fopenmp
CLUSTERING_OBJECTS = single_use_barrier.o timespec_functions.o
NIincludes = -I/usr/local/natinst/nidaqmxbase/include
NIlibs=-lnidaqmxbase

#all: clustering_distribution simple_task synthetic_task
#all: clustering_distribution synthetic_task synthetic_task_gd synthetic_task_gd_extra
all: clustering_distribution

synthetic_task: synthetic_task.cpp
	$(CC) $(FLAGS) -fopenmp synthetic_task.cpp sharedMem.o task.o task_manager.o bar.o schedule.o taskData.o -o synthetic_task $(LIBS)

synthetic_task_gd: synthetic_task_gd.cpp
	$(CC) $(FLAGS) -fopenmp synthetic_task_gd.cpp sharedMem.o task_manager.o task.o bar.o  -o st_gd $(LIBS)

synthetic_task_gd_extra: synthetic_task_gd_extra.cpp
	$(CC) $(FLAGS) -fopenmp synthetic_task_gd_extra.cpp sharedMem.o task_manager.o task.o bar.o  -o st_extra $(LIBS)

bar.o: bar.cpp
	$(CC) $(FLAGS) -c bar.cpp
#synthetic_task_utilization: synthetic_task.cpp
#       $(CC) $(FLAGS) -fopenmp synthetic_task.cpp utilization_calculator.o -o synthetic_task_utilization $(LIBS)
#
#       simple_task: simple_task.cpp
#               $(CC) $(FLAGS) -fopenmp simple_task.cpp mode.o sharedMem.o task_manager.o bar.o -o simple_task $(LIBS)
#               #simple_task_utilization: simple_task.cpp
#       $(CC) $(FLAGS) -fopenmp simple_task.cpp utilization_calculator.o -o simple_task_utilization $(LIBS)
#
clustering_distribution: libclustering.a sharedMem.o schedule.o scheduler.o task.o taskData.o task_manager.o bar.o clustering_launcher synthetic_task james

libclustering.a: $(CLUSTERING_OBJECTS)
	ar rcsf libclustering.a $(CLUSTERING_OBJECTS)

task.o: task.cpp
	$(CC) $(FLAGS) -c task.cpp

task_manager.o: schedule.cpp schedule.cpp sharedMem.cpp task_manager.cpp
	$(CC) $(FLAGS) -fopenmp -c task_manager.cpp

single_use_barrier.o: single_use_barrier.cpp
	$(CC) $(FLAGS) -c single_use_barrier.cpp

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

#mode.o: sharedMem.cpp mode.cpp
#	$(CC) $(FLAGS) -c mode.cpp


clustering_launcher: clustering_launcher.cpp
	$(CC) $(FLAGS) taskData.o schedule.o scheduler.o sharedMem.o clustering_launcher.cpp -o clustering_launcher $(LIBS)

james: james.cpp task_manager.o
	$(CC) $(FLAGS) james.cpp sharedMem.o scheduler.o schedule.o taskData.o task.o task_manager.o bar.o -o james $(LIBS)
	cp james phil

clean:
	rm -f *.o  *.pyc libclustering.a clustering_launcher synthetic_task_gd synthetic_task synthetic_task_gd_extra mixed_crit_test james phil
