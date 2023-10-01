#ifndef _TASKDATA_H
#define  _TASKDATA_H
#include <algorithm>
#include <assert.h>
#include <stdlib.h> 
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include "timespec_functions.h"
#include "include.h"

//extern const int NUMCPUS = 31;//There is always 1 CPU set aside for the scheduler object.
//extern const int MAXTASKS = 10;
//extern const int MAXMODES = 10;

class TaskData{
private:
	static int counter;
	int index; //unique identifier
	//int total_CPUs;
	bool changeable;
	bool can_reschedule;
	int num_adaptations;

	//These are read in. James 9/7/18
	double elasticity;	
	int num_modes;
	timespec work[MAXMODES];
	timespec span[MAXMODES];
	timespec period[MAXMODES];
	int CPUs[MAXMODES];

	//These are computed.
	double max_utilization;
	double min_utilization;
	int max_CPUs;
	int min_CPUs;

	int CPUs_gained;

	double practical_max_utilization;
	int practical_max_CPUs;	

	int current_lowest_CPU;

	double percentage_workload;

	timespec current_period;
	timespec current_work;
	timespec current_span;
	double current_utilization;
	int current_CPUs;
	int previous_CPUs;

	int permanent_CPU;

	int current_mode;
	timespec max_work;

	int active[NUMCPUS+1];
	int passive[NUMCPUS+1];

	int give[MAXTASKS];
	bool transfer[MAXTASKS][NUMCPUS+1];
	bool receive[MAXTASKS][NUMCPUS+1];
public:
	TaskData(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_) : index(counter++), changeable(true), can_reschedule(false), num_adaptations(0),  elasticity(elasticity_), num_modes(num_modes_), max_utilization(0), max_CPUs(0), min_CPUs(NUMCPUS),  CPUs_gained(0), practical_max_utilization(max_utilization),  practical_max_CPUs(max_CPUs), current_lowest_CPU(-1), percentage_workload(1.0), current_period({0,0}), current_work({0,0}), current_span({0,0}), current_utilization(0.0), current_CPUs(0), previous_CPUs(0), permanent_CPU(-1), current_mode(0), max_work({0,0})
	{
		if(num_modes > MAXMODES)
		{
			fprintf(stderr, "ERROR: No task can have more than %d modes.\n",MAXMODES);
			kill(0, SIGTERM);
		}

		for(int i=0; i<num_modes; i++)
		{
			work[i] = *(work_+i); 
			span[i] = *(span_+i); 
			period[i] = *(period_+i); 
		}			
		for(int i=0; i<num_modes; i++)
		{	
			std::cout << work[i] << " " << span[i] << " " << period[i] << std::endl;	
		}	

		timespec numerator;
		timespec denominator;

		for(int i=0; i<num_modes; i++)
		{   
			if(work[i]/period[i] > max_utilization)
			{
				max_utilization=work[i]/period[i];
			}
			ts_diff(work[i],span[i],numerator);
			ts_diff(period[i],span[i],denominator);
			CPUs[i] = (int) ceil(numerator/denominator);
		
			if(CPUs[i] > max_CPUs)
			{
				max_CPUs = CPUs[i];
			}

			if(CPUs[i] < min_CPUs)
			{
				min_CPUs = CPUs[i];
			}

			if(work[i] > max_work)
			{
				max_work = work[i];
			}
		}
		current_CPUs = min_CPUs;

		for(int i=0; i<MAXTASKS; i++)
		{
			
			give[i]=0;
			for(int j=1; j<=NUMCPUS; j++)
			{
				transfer[i][j] = false;
				receive[i][j] = false;
				active[i]=false;
				passive[i]=false;
			}
		}
	}

	~TaskData()
	{
	}

	int get_num_modes();	
	int get_index();
	timespec get_current_span();
	double get_elasticity();
	double get_percentage_workload();
	bool get_changeable();
	
	double get_max_utilization();
	double get_min_utilization();
	int get_max_CPUs();
	int get_min_CPUs();

	timespec get_max_work();

	double get_practical_max_utilization();
	int get_practical_max_CPUs();
	void set_practical_max_CPUs(int new_value);

	timespec get_current_period();
	timespec get_current_work();
    double get_current_utilization();
	int get_current_CPUs();
	int get_current_lowest_CPU();
	
	int get_CPUs_gained();
	void set_CPUs_gained(int new_CPUs_gained);
	int get_previous_CPUs();
	void set_previous_CPUs(int new_prev);

	void set_current_mode(int new_mode, bool disable);

	//void set_current_work(timespec new_work, bool disable);
	//void set_current_span(timespec new_work, bool disable);
	//void set_current_period(timespec new_period, bool disable);

	//void set_current_mode(int new_mode, bool disable);
	int get_current_mode();

	//void set_current_CPUs(int new_CPUs);
	void reset_changeable();
	void set_current_lowest_CPU(int _lowest);

	void update_give(int index, int value);
	int gives(int index);

	bool transfers(int task, int CPU);
	void set_transfer(int task, int CPU, bool value);

	bool receives(int task, int CPU);
	void set_receive(int task, int CPU, bool value);

	int get_permanent_CPU();
	void set_permanent_CPU(int perm);
	//assert deadline==current_period;
	
	void set_active(int i);
	void clr_active(int i);
	void set_passive(int i);
	void clr_passive(int i);
	bool get_active(int i);
	bool get_passive(int i);

	int get_num_adaptations();
	void set_num_adaptations(int new_num);

	timespec get_work(int index);
	timespec get_span(int index);
	timespec get_period(int index);
	int get_CPUs(int index);
};

#endif
