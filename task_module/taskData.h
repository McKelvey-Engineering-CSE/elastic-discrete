#ifndef _TASKDATA_H
#define  _TASKDATA_H

/*************************************************************************

taskData.h

This object returns a bunch of data about a given task... need to look at code
more to come up with a better description than that


Class : TaskData

		Class that stores information related to a given task, as well as providing
		getters and setters to modify those data points/values
        
**************************************************************************/

#include <algorithm>
#include <assert.h>
#include <stdlib.h> 
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <errno.h>
#include <string.h>
#include <sys/msg.h>
#include <array>
#include <tuple>
#include <functional>
#include "timespec_functions.h"
#include "include.h"
#include "print_module.h"

struct queue_one_message {
    long    mtype;
	__uint128_t tasks_giving_processors;
};

struct queue_two_message {
	long    mtype;
	long int   giving_task;
	long int   processor_type;
	__uint128_t processors;
};

struct queue_three_message {
	long    mtype;
	long int   task_index;
	long int   processor_type;
	long int   processor_ct;
};

class TaskData{

private:

	//variable to track if this task is a pure A processor task
	bool is_pure_A_task = true;

	//variable to track if the task is combinatorially elastic
	bool combinatorially_elastic = false;

	static int counter;
	int index; //unique identifier
	
	bool changeable = true;
	bool can_reschedule;
	int num_adaptations;

	//These are read in. James 9/7/18
	double elasticity;	
	int num_modes;
	timespec work[MAXMODES];
	timespec span[MAXMODES];
	timespec period[MAXMODES];
	int processors_A[MAXMODES];

	timespec processor_B_work[MAXMODES];
	timespec processor_B_span[MAXMODES];
	timespec processor_B_period[MAXMODES];
	int processors_B[MAXMODES];

	timespec processor_C_work[MAXMODES];
	timespec processor_C_span[MAXMODES];
	timespec processor_C_period[MAXMODES];
	int processors_C[MAXMODES];

	timespec processor_D_work[MAXMODES];
	timespec processor_D_span[MAXMODES];
	timespec processor_D_period[MAXMODES];
	int processors_D[MAXMODES];

	int owning_modes[MAXMODES];

	//simple map to assign multiple possible
	//mode configurations to a single mode
	int mode_map[MAXMODES];
	int real_current_mode;

	//These are computed.
	double max_utilization;
	int max_processors_A;
	int min_processors_A;

	int max_processors_B;
	int min_processors_B;

	int max_processors_C;
	int min_processors_C;

	int max_processors_D;
	int min_processors_D;

	int processors_A_gained;
	int processors_B_gained;
	int processors_C_gained;
	int processors_D_gained;

	int practical_max_processors_A;	
	int current_lowest_processor_A;

	int practical_max_processors_B;	
	int current_lowest_processor_B = 0;

	int practical_max_processors_C;	
	int current_lowest_processor_C;

	int practical_max_processors_D;	
	int current_lowest_processor_D = 0;

	double percentage_workload;

	timespec current_period;
	timespec current_work;
	timespec current_span;
	double current_utilization;

	int current_processors_A;
	int previous_processors_A;

	int current_processors_B;
	int previous_processors_B;

	int current_processors_C;
	int previous_processors_C;

	int current_processors_D;
	int previous_processors_D;

	int permanent_processor_A = -1;
	int permanent_processor_B = -1;
	int permanent_processor_C = -1;
	int permanent_processor_D = -1;

	int permanent_processor_core = -1;

	//if the index of the processor is -1, then the processor is not equivalent to any other processor
	int processors_equivalent_to_A[4] = {0, 0, 0, 0};

	int permanent_processor_index; //index of the processor that is permanently assigned to this task (0 is A, 1 is B, 2 is C, 3 is D)

	int current_logical_mode;
	timespec max_work;

	int current_virtual_mode;

	bool cooperative_bool = true;

	int number_of_modes = 0;

	//processor B mask
	__uint128_t processor_B_mask = 0;

	//processor A mask
	__uint128_t processor_A_mask = 0;

	//processor C mask
	__uint128_t processor_C_mask = 0;

	//processor D mask
	__uint128_t processor_D_mask = 0;

	//updated variables
	bool mode_transitioned = false;

	//these variables are set by the scheduler to denote
	//how many of our resources we are supposed to return
	int processors_A_to_return = 0;
	int processors_B_to_return = 0;
	int processors_C_to_return = 0;
	int processors_D_to_return = 0;

	//these denote the number of tasks we are looking for
	//when we are collecting our resources. The scheduler 
	//will use these more than we will as tasks
	int other_tasks_giving_processors_A = 0;
	int other_tasks_giving_processors_B = 0;
	int other_tasks_giving_processors_C = 0;
	int other_tasks_giving_processors_D = 0;

	//these are the indicies of processors we are gaining
	//and which processors they came from

	//the position MAXTASKS + 1 is the free resource pool
	//it is treated as a task for no reason other than 
	//consistency
	int processors_A_granted_from_other_tasks [MAXTASKS + 1][NUM_PROCESSOR_A + 1];
	int processors_B_granted_from_other_tasks [MAXTASKS + 1][NUM_PROCESSOR_B + 1];
	int processors_C_granted_from_other_tasks [MAXTASKS + 1][NUM_PROCESSOR_C + 1];
	int processors_D_granted_from_other_tasks [MAXTASKS + 1][NUM_PROCESSOR_D + 1];

	//message queue ids used for resource exchange
	int queue_one;
	int queue_two;
	int queue_three;

	//these structures are used to track the state of the message queues
	//and messages we have received from other tasks
	__uint128_t tasks_giving_processors = 0;
	__uint128_t processors_A_received = 0;
	__uint128_t processors_B_received = 0;
	__uint128_t processors_C_received = 0;
	__uint128_t processors_D_received = 0;

	int previous_mode = 0;

	int modes_originally_passed;

	int previous_permanent_processor_index = -1;

public:

	TaskData(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * processor_B_work_, timespec * processor_B_span_, timespec * processor_B_period_, timespec * processor_C_work_, timespec * processor_C_span_, timespec * processor_C_period_, timespec * processor_D_work_, timespec * processor_D_span_, timespec * processor_D_period_, bool safe, std::tuple<int, float> equivalent_vector[4], bool print);

	TaskData();

	TaskData(bool free_resources);

	~TaskData();

	int get_index();
	double get_elasticity();
	double get_percentage_workload();
	bool get_changeable();
	
	double get_max_utilization();
	int get_max_processors_A();
	int get_min_processors_A();

	int get_real_mode(int mode);
	int get_number_of_modes();

	int get_practical_max_processors_A();
	void set_practical_max_processors_A(int new_value);

	timespec get_current_period();
	timespec get_current_work();
	timespec get_current_span();

	int get_current_processors_A();
	int get_current_lowest_processor_A();
	
	int get_processors_A_gained();
	void set_processors_A_gained(int new_processors_A_gained);

	void set_current_virtual_mode(int new_mode, bool disable);
	int get_current_virtual_mode();

	void reset_changeable();
	void set_current_lowest_processor_A(int _lowest);

	int get_permanent_processor_A();
	void set_permanent_processor_A(int perm);
	
	int get_num_adaptations();
	void set_num_adaptations(int new_num);

	timespec get_work(int index);
	timespec get_span(int index);
	timespec get_period(int index);
	int get_processors_A(int index);

	//processor B functions that can be compiled regardless of compiler
	timespec get_processor_B_work(int index);
	timespec get_processor_B_span(int index);
	timespec get_processor_B_period(int index);

	int get_processors_B(int index);
	int get_max_processors_B();
	int get_min_processors_B();
	int get_current_processors_B();

	//processor C functions that can be compiled regardless of compiler
	timespec get_processor_C_work(int index);
	timespec get_processor_C_span(int index);
	timespec get_processor_C_period(int index);
	int get_processors_C(int index);

	//processor D functions that can be compiled regardless of compiler
	timespec get_processor_D_work(int index);
	timespec get_processor_D_span(int index);
	timespec get_processor_D_period(int index);
	int get_processors_D(int index);

	int get_max_processors_C();
	int get_min_processors_C();
	int get_current_processors_C();

	int get_max_processors_D();
	int get_min_processors_D();
	int get_current_processors_D();

	void set_practical_max_processors_B(int new_value);
	int get_practical_max_processors_B();
	void set_current_lowest_processor_B(int _lowest);
	int get_current_lowest_processor_B();

	void set_practical_max_processors_C(int new_value);
	int get_practical_max_processors_C();
	void set_current_lowest_processor_C(int _lowest);
	int get_current_lowest_processor_C();

	void set_practical_max_processors_D(int new_value);
	int get_practical_max_processors_D();
	void set_current_lowest_processor_D(int _lowest);
	int get_current_lowest_processor_D();

	int get_processors_B_gained();
	void set_processors_B_gained(int new_processors_B_gained);

	int get_processors_C_gained();
	void set_processors_C_gained(int new_processors_C_gained);

	int get_processors_D_gained();
	void set_processors_D_gained(int new_processors_D_gained);

	bool pure_A_task();

	//reworking all the processor A and processor B handoff functions
	//NOTE: all return functions will work from the 
	//highest processor A/processor B unit we have down until we run
	//out of processor A/processor B to return
	void set_processors_A_change(int num_processors_A_to_return);

	void set_processors_B_change(int num_processors_B_to_return);

	void set_processors_C_change(int num_processors_C_to_return);

	void set_processors_D_change(int num_processors_D_to_return);

	int get_processors_A_change();

	int get_processors_B_change();

	int get_processors_C_change();

	int get_processors_D_change();

	int get_real_current_mode();
	
	void set_real_current_mode(int new_mode, bool disable);

	//function to check if this task has transitioned
	//to a new mode yet
	bool check_mode_transition();

	void set_mode_transition(bool state);

	int get_original_modes_passed();

	//functions to work with static vector of processor A indices
	int pop_back_processor_A();

	int push_back_processor_A(int index);

	int pop_back_processor_B();

	int push_back_processor_B(int index);

	int pop_back_processor_C();

	int push_back_processor_C(int index);

	int pop_back_processor_D();

	int push_back_processor_D(int index);

	int get_processor_A_at_index(int index);

	std::vector<int> get_processor_A_owned_by_process();

	std::vector<int> get_processor_B_owned_by_process();

	std::vector<int> get_processor_C_owned_by_process();

	std::vector<int> get_processor_D_owned_by_process();

	//retrieve the number of processor A or processor B we have been given	
	std::vector<std::pair<int, std::vector<int>>> get_processors_A_granted_from_other_tasks();

	std::vector<std::pair<int, std::vector<int>>> get_processors_B_granted_from_other_tasks();

	std::vector<std::pair<int, std::vector<int>>> get_processors_C_granted_from_other_tasks();

	std::vector<std::pair<int, std::vector<int>>> get_processors_D_granted_from_other_tasks();

	//give processor A or processor B to another task
	void set_processors_A_to_send_to_other_processes(std::pair<int, int> entry);

	void set_processors_B_to_send_to_other_processes(std::pair<int, int> entry);

	void set_processors_C_to_send_to_other_processes(std::pair<int, int> entry);

	void set_processors_D_to_send_to_other_processes(std::pair<int, int> entry);

	//make a function which clears these vectors like they are cleared in the constructor
	void clear_processors_A_granted_from_other_tasks();

	void clear_processors_B_granted_from_other_tasks();

	void clear_processors_C_granted_from_other_tasks();

	void clear_processors_D_granted_from_other_tasks();

	__uint128_t get_processor_A_mask();

	__uint128_t get_processor_B_mask();

	__uint128_t get_processor_C_mask();

	__uint128_t get_processor_D_mask();

	//function to get and set combinationally elastic
	bool is_combinatorially_elastic();

	void set_cooperative(bool state);

	bool cooperative();

	//new message based resource passing functions
	void set_processors_to_send_to_other_processes(int task_to_send_to, int processor_type, int processor_ct);

	void set_tasks_to_wait_on(__uint128_t task_mask);

	void start_transition();

	bool get_processors_granted_from_other_tasks();

	void acquire_all_processors();

	void give_processors_to_other_tasks();

	void reset_mode_to_previous();

	int get_permanent_processor_index();

	int get_permanent_processor_core();

	void set_permanent_processor_index(int index);
	
	void set_permanent_processor_core(int core);

	int get_previous_permanent_processor_index();

	void acknowledge_permanent_processor_switch();

	void elect_permanent_processor();

};

#endif
