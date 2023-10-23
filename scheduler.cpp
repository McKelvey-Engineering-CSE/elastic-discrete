#include "scheduler.h"
#include "taskData.h"

#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cerrno>
#include <float.h>

#ifdef SCHED_PAIR_HEAP

	/*October 23 2023 - moved class to be nested inside in-case it actually is needed
	in the future. I doubt this, and am leaving it non-compiling for the meantime.
	If it turns out the inner class "sched_pair" is needed, these functions should be
	moved back out of the class so that we can use references.

	(Also, make them idiomatic in the future if we actually need them)
	*/
	bool Scheduler::sched_pair::operator>(const Scheduler::sched_pair& rhs){
	
	return(this->weight > rhs.weight);

	}

	bool Scheduler::sched_pair::operator<(const Scheduler::sched_pair& rhs){
	
	return(this->weight < rhs.weight);

	}

	bool Scheduler::sched_pair::operator==(const Scheduler::sched_pair& rhs){
	
	return(this->weight == rhs.weight);

	}

	bool Scheduler::sched_pair::operator<=(const Scheduler::sched_pair& rhs){
	
	return (this->weight < rhs.weight) || (this->weight == rhs.weight);

	}

	bool Scheduler::sched_pair::operator>=(const Scheduler::sched_pair& rhs){
	
	return (this->weight > rhs.weight) || (this->weight == rhs.weight);

	}

	bool Scheduler::sched_pair::operator!=(const Scheduler::sched_pair& rhs){
	
	return(!(this->weight == rhs.weight));

	}

#endif

class Schedule * Scheduler::get_schedule(){
	return &schedule;
}

TaskData * Scheduler::add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_){
	
	return schedule.add_task(elasticity_, num_modes_, work_, span_, period_);
}

//Implement scheduling algorithm
void Scheduler::do_schedule(){

	//First time through Make sure we have enough CPUs in the system and determine practical max for each task.	
	if(first_time)
	{
		int min_required = 0;

		//Determine minimum required processors
		for(int i=0; i<schedule.count(); i++)
		{
			min_required += (schedule.get_task(i))->get_min_CPUs();
			(schedule.get_task(i))->set_CPUs_gained(0);
		}

		//Determine the practical maximum. This is how many are left after each task has been given its minimum.
		for(int i=0; i<schedule.count(); i++)
		{
			if((NUMCPUS - min_required + (schedule.get_task(i))->get_min_CPUs()) < (schedule.get_task(i))->get_max_CPUs())
			{
				(schedule.get_task(i))->set_practical_max_CPUs( NUMCPUS - min_required + (schedule.get_task(i))->get_min_CPUs());
			}
			else
			{
				(schedule.get_task(i))->set_practical_max_CPUs((schedule.get_task(i))->get_max_CPUs());
			}
		}
	}

	//James Start 10/4/18
	
	//Assign initial  score of infinity.
	for(int l=0; l<=(schedule.count()); l++)
	{
		for(int d=0; d<=NUMCPUS; d++)
		{
			DP[d][l].first=std::numeric_limits<double>::max();
		}
	}
	
	//Consider scheduling on d processors.
	for(int d=1; d<=NUMCPUS; d++)
	{
		//Consider scheduling first l tasks.
		for(int l=1; l<=schedule.count(); l++)
		{

			//TODO: FIGURE OUT WHAT TO DO IN ORDER TO SKIP A TASK!!
			if(!(schedule.get_task(l-1))->get_changeable())
			{
				if(l==1)
				{
					std::vector<int> temp;
					temp.push_back((schedule.get_task(l-1))->get_current_mode());
					DP[d][l]=std::make_pair(std::numeric_limits<double>::max(),temp);
				}
				else
				{
					std::vector<int> temp = DP[d-(schedule.get_task(l-1)->get_CPUs((schedule.get_task(l-1))->get_current_mode()))][l-1].second;
					temp.push_back((schedule.get_task(l-1))->get_current_mode());
                	DP[d][l]=std::make_pair(DP[d-(schedule.get_task(l-1)->get_CPUs((schedule.get_task(l-1))->get_current_mode()))][l-1].first,temp);
				}
			}
			else
			{
				//Assign the minimum to be infinity.
				double MIN=std::numeric_limits<double>::max();
				int selection=-1;

				//Consider mode j
				for(int j=0; j<(schedule.get_task(l-1))->get_num_modes(); j++)
				{
					//Make sure there are enough remaining processors for the mode we're considering.
					if(d-(schedule.get_task(l-1)->get_CPUs(j)) >= 0)
					{
						//Special case for the first task. Assign MIN score of just this task.
						//We chose mode j.
						if(l==1 && (1.0/(schedule.get_task(l-1))->get_elasticity()*(std::pow((schedule.get_task(l-1))->get_max_utilization()-((schedule.get_task(l-1)->get_work(j))/(schedule.get_task(l-1)->get_period(j))),2))) < MIN)
						{
							MIN= (1.0/(schedule.get_task(l-1))->get_elasticity()*(std::pow((schedule.get_task(l-1))->get_max_utilization()-((schedule.get_task(l-1)->get_work(j))/(schedule.get_task(l-1)->get_period(j))),2)));
							selection=j;
						}
						//Otherwise must consider score from first prior tasks when finding the minimum.
						//We chose mode j.
						else if(DP[(d-(schedule.get_task(l-1)->get_CPUs(j)))][l-1].first + (1.0/(schedule.get_task(l-1))->get_elasticity()*(std::pow((schedule.get_task(l-1))->get_max_utilization()-((schedule.get_task(l-1)->get_work(j))/(schedule.get_task(l-1)->get_period(j))),2))) < MIN)
						{
							MIN = DP[(d-(schedule.get_task(l-1)->get_CPUs(j)))][l-1].first + (1.0/(schedule.get_task(l-1))->get_elasticity()*(std::pow((schedule.get_task(l-1))->get_max_utilization()-((schedule.get_task(l-1)->get_work(j))/(schedule.get_task(l-1)->get_period(j))),2)));
							selection=j;
						}//endif
					}//endif	
				}//endfor j

				//Update DP.				
				if(d-(schedule.get_task(l-1)->get_CPUs(selection)) >= 0)
				{
					std::vector<int> temp = DP[d-(schedule.get_task(l-1)->get_CPUs(selection))][l-1].second;
					temp.push_back(selection);
					DP[d][l]=std::make_pair(std::min(MIN,DP[d-1][l].first),temp);
				}
			}
		}//endfor l
	}//endfor d


	//James Stop 10/4/18

	std::cout << "Got this allocation: " << DP[NUMCPUS][schedule.count()].first << " ";
	for(unsigned int i=0; i<DP[NUMCPUS][schedule.count()].second.size(); i++)
	{
		std::cout << DP[NUMCPUS][schedule.count()].second[i];
	}
	std::cout << "\n";

	for(int i=0; i<schedule.count(); i++)
	{
		(schedule.get_task(i))->set_current_mode(DP[NUMCPUS][schedule.count()].second[i],false);
	}

//give_cpus:
	//First allocate from CPU1 and go up from there.
	if(first_time)
	{
		int next_CPU=1;
		//Actually assign CPUs to tasks. Start with 1.
		for(int i=0; i<schedule.count(); i++)
		{
			if((schedule.get_task(i))->get_current_lowest_CPU() > 0)
			{
				std::cerr << "Error in task " << i << ": all tasks should have had lowest CPU cleared.\n";
				killpg(process_group, SIGKILL);
                return;
			}

			(schedule.get_task(i))->set_current_lowest_CPU(next_CPU);
			next_CPU += (schedule.get_task(i))->get_current_CPUs();

			if(next_CPU > num_CPUs+1)
			{
				std::cerr << "Error in task " << i << ": too many CPUs have been allocated." << next_CPU << " " << num_CPUs<< " \n";
				killpg(process_group, SIGKILL);
				return;
			}		
		}
	}
	//Transfer as efficiently as possible.
	else
	{
		for(int i=0; i<schedule.count(); i++)
		{
		
			int gained = (schedule.get_task(i))->get_current_CPUs() - (schedule.get_task(i))->get_previous_CPUs(); 
			(schedule.get_task(i))->set_CPUs_gained(gained);
		}

		//First determine how many CPUs each task gets from each other task.
		for(int i=0; i<schedule.count(); i++)
		{
			for(int j=i+1; j<schedule.count(); j++)
			{
				if((schedule.get_task(i))->get_CPUs_gained() > 0 && (schedule.get_task(j))->get_CPUs_gained() < 0) 	
				{
					int difference = abs((schedule.get_task(i))->get_CPUs_gained()) - abs((schedule.get_task(j))->get_CPUs_gained());
					
					if(difference >= 0)
					{
						int amount = abs((schedule.get_task(j))->get_CPUs_gained());
						(schedule.get_task(i))->set_CPUs_gained((schedule.get_task(i))->get_CPUs_gained() - amount);
						(schedule.get_task(j))->set_CPUs_gained((schedule.get_task(j))->get_CPUs_gained() + amount);
						(schedule.get_task(i))->update_give(j,-1*amount);
						(schedule.get_task(j))->update_give(i,amount);
					}
					else
					{
						int amount = abs((schedule.get_task(i))->get_CPUs_gained());
						(schedule.get_task(i))->set_CPUs_gained((schedule.get_task(i))->get_CPUs_gained() - amount);
						(schedule.get_task(j))->set_CPUs_gained((schedule.get_task(j))->get_CPUs_gained() + amount);
						(schedule.get_task(i))->update_give(j,-1*amount);
						(schedule.get_task(j))->update_give(i,amount);

					}
				}
				else if((schedule.get_task(j))->get_CPUs_gained() > 0 && (schedule.get_task(i))->get_CPUs_gained() < 0)
				{
					int difference = abs((schedule.get_task(j))->get_CPUs_gained()) - abs((schedule.get_task(i))->get_CPUs_gained());
					
					if(difference >= 0){
						int amount = abs((schedule.get_task(i))->get_CPUs_gained());
						(schedule.get_task(j))->set_CPUs_gained((schedule.get_task(j))->get_CPUs_gained() - amount);
						(schedule.get_task(i))->set_CPUs_gained((schedule.get_task(i))->get_CPUs_gained() + amount);
						(schedule.get_task(j))->update_give(i,-1*amount);
						(schedule.get_task(i))->update_give(j,amount);
					}
					else{
						int amount = abs((schedule.get_task(j))->get_CPUs_gained());
						(schedule.get_task(j))->set_CPUs_gained((schedule.get_task(j))->get_CPUs_gained() - amount);
						(schedule.get_task(i))->set_CPUs_gained((schedule.get_task(i))->get_CPUs_gained() + amount);
						(schedule.get_task(j))->update_give(i,-1*amount);
						(schedule.get_task(i))->update_give(j,amount);
					}
				}
			}
		}
		//Now determine which CPUs are transfered.
		for(int i=0; i<schedule.count(); i++)
                {
			for(int j=i+1; j<schedule.count(); j++)
                        {		
				cpu_set_t overlap;
				if((schedule.get_task(i))->gives(j) > 0)
				{
					int amount_gives = (schedule.get_task(i))->gives(j);
				
					for(int t=1; t<=NUMCPUS; t++){
						if(schedule.get_task(i)->get_active(t) && schedule.get_task(j)->get_passive(t))
						{
							CPU_SET(t,&overlap);
						}
					}
	
					int amount_overlap = CPU_COUNT(&overlap);
					if(amount_overlap > 0)
					{
						for(int k=1; k<=NUMCPUS; k++)
						{
							if(CPU_ISSET(k,&overlap))
							{
								bool used = false;
								for(int l=0; l<schedule.count(); l++)
								{
										if(schedule.get_task(i)->transfers(l,k))
										{
												used=true;
										}
								}

								if(!used && k!=(schedule.get_task(i))->get_permanent_CPU() && !(schedule.get_task(i))->transfers(j,k) )
								{
									std::cerr << "Task " << i << " should be sending CPU " << k << " to task " << j << ".\n";
									
									(schedule.get_task(i))->set_transfer(j,k,true);

									(schedule.get_task(j))->set_receive(i,k,true);
									
									amount_gives--;

									if(amount_gives == 0)
									{
										break;
									}
								}
							}
						}
					}
					if(amount_gives > 0)
					{
						for(int k=NUMCPUS; k >=1; k--)
						{
							if(schedule.get_task(i)->get_active(k))
							{
								bool used = false;
								for(int l=0; l<schedule.count(); l++)
								{
										if(schedule.get_task(i)->transfers(l,k))
										{
												used=true;
										}
								}   
	
								if(!used && k!=(schedule.get_task(i))->get_permanent_CPU() && !(schedule.get_task(i))->transfers(j,k))
								{
									std::cerr << "Task " << i << " should be sending CPU " << k << " to task " << j << ".\n";

									(schedule.get_task(i))->set_transfer(j,k,true);
									
									(schedule.get_task(j))->set_receive(i,k,true);

									amount_gives--;
									if(amount_gives == 0)
									{
										break;
									}
                                }

							}
						}
					}
				}
				else if((schedule.get_task(i))->gives(j) < 0)
				{	
					int amount_gives = (schedule.get_task(j))->gives(i);
						
					cpu_set_t first;
					cpu_set_t second;			
					CPU_ZERO(&first);
					CPU_ZERO(&second);
					CPU_ZERO(&overlap);

					for(int t=1; t<=NUMCPUS; t++)
					{
						if(schedule.get_task(j)->get_active(t))
						{
							CPU_SET(t, &first);

						}

						if(schedule.get_task(i)->get_passive(t))
						{
							CPU_SET(t, &second);
				
						}
						
						

                    }
	
					CPU_AND(&overlap, &first, &second);

					int amount_overlap = CPU_COUNT(&overlap);

					if(amount_overlap > 0)
					{
						for(int k=1; k<=NUMCPUS; k++)
						{
							if(CPU_ISSET(k,&overlap))
							{
								bool used = false;
								for(int l=0; l<schedule.count(); l++)
								{
									if(schedule.get_task(j)->transfers(l,k))
									{
										used=true;	
									}
								}

								if(!used &&  k!=(schedule.get_task(j))->get_permanent_CPU() && !(schedule.get_task(j))->transfers(i,k))
								{
									
									(schedule.get_task(j))->set_transfer(i,k,true);
									
									(schedule.get_task(i))->set_receive(j,k,true);
									
									std::cerr << "Task " << j << "  should be sending CPU " << k << " to task " << i << ".\n";

									amount_gives--;

									if(amount_gives == 0)
									{
										break;
									}
								}
							}
						}
					}
					if(amount_gives > 0)
					{
						for(int k=NUMCPUS; k >=1; k--)
						{
							if( schedule.get_task(j)->get_active(k))
							{
								bool used = false;

								for(int l=0; l<schedule.count(); l++)
								{
									if(schedule.get_task(j)->transfers(l,k))
									{
											used=true;
									}	
								}
								if(!used && k!=(schedule.get_task(j))->get_permanent_CPU() && !(schedule.get_task(j))->transfers(i,k))
								{
									(schedule.get_task(j))->set_transfer(i,k,true);
									
									(schedule.get_task(i))->set_receive(j,k,true);
									
									std::cerr << "Task " << j << "  should be sending CPU " << k << " to task " << i << ".\n";

									amount_gives--;

									if(amount_gives == 0)
									{
										break;
									}
                                }
							}
						}
					}
				}
			}			
		}
		
	}
	first_time = false;
}
