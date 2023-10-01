#include "scheduler.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <float.h>
#include "taskData.h"

bool operator>(const sched_pair& lhs, const sched_pair& rhs){
  if(lhs.weight > rhs.weight){
    return true;
  }

  return false;
}

bool operator<(const sched_pair& lhs, const sched_pair& rhs){
  if(lhs.weight < rhs.weight){
    return true;
  } 
  
  return false;
}

bool operator==(const sched_pair& lhs, const sched_pair& rhs){
  if(lhs.weight == rhs.weight){
    return true;
  }

  return false;
}

bool operator<=(const sched_pair& lhs, const sched_pair& rhs){
  if((lhs < rhs) || (lhs == rhs)){
    return true;
  }
  return false;
}

bool operator>=(const sched_pair& lhs, const sched_pair& rhs){
  if((lhs > rhs) || (lhs == rhs)){
    return true;
  }
  return false;
}

bool operator!=(const sched_pair& lhs, const sched_pair& rhs){
  if(!(lhs == rhs)){
    return true;
  }
  return false;
}

class Schedule * Scheduler::get_schedule(){
	return &schedule;
}
TaskData * Scheduler::add_task (double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_){
	
	//Reserve enough space for algorithm
	/*std::vector<std::vector<double>> weights;
	std::vector<double> task;
	task.reserve(num_modes_);
	weights.push_back(std::make_pair(task,""));
	*/
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
	

	//fprintf(stderr,"Got this allocation: %f %s \n",DP[NUMCPUS][schedule.count()].first,DP[NUMCPUS][schedule.count()].second.c_str());	
	fprintf(stderr, "Got this allocation: %f ",DP[NUMCPUS][schedule.count()].first);
	for(unsigned int i=0; i<DP[NUMCPUS][schedule.count()].second.size(); i++)
	{
		fprintf(stderr, "%d ", DP[NUMCPUS][schedule.count()].second[i]);
	}
	fprintf(stderr, "\n");

	for(int i=0; i<schedule.count(); i++)
	{
		//int temp = std::atoi(&(DP[NUMCPUS][schedule.count()].second.c_str()[i]));
		(schedule.get_task(i))->set_current_mode(DP[NUMCPUS][schedule.count()].second[i],false);
	}



	/*for(int d=1; d<=NUMCPUS; d++)
	{
		std::cout << "CPUS: " << d;
		for(int l=1; l<=(schedule.count()); l++)
		{
			std::cout << " Task: " << l << " Score: " << DP[d][l].first;// << std::endl;
		}
		std::cout << std::endl;
	}*/

	//fprintf(stderr,"Got this allocation: %s %d %d \n",DP[NUMCPUS][num_tasks].second.c_str(),NUMCPUS,num_tasks);	
/*	int m_used = 0;
	
	//Make sure taskset is schedulable. Begin with giving all tasks their minimum number of CPUs.
	for(int i=0; i<schedule.count(); i++)
	{
		if(!first_time)
		{
			if((schedule.get_task(i))->get_changeable())
			{	
				(schedule.get_task(i))->set_previous_CPUs((schedule.get_task(i))->get_current_CPUs());
			}
			//fprintf(stderr, "task %d previous_CPUs = %d\n",i,(schedule.get_task(i))->get_current_CPUs());
		}

		//If tasks have been set to not change, don't worry about minimum. Instead use current.
		if(!(schedule.get_task(i))->get_changeable())
		{
			m_used += (schedule.get_task(i))->get_current_CPUs();
		}
		else{
			//Minimum is used in schedulability test	
			int temp = (schedule.get_task(i))->get_min_CPUs();

			m_used += temp;
			(schedule.get_task(i))->set_current_CPUs(temp);
		}

		for(int j=0; j<schedule.count(); j++)
		{
			for(int k=1; k<=NUMCPUS; k++)
			{
			(schedule.get_task(i))->set_transfer(j,k,false);
                        (schedule.get_task(i))->set_receive(j,k,false);
			}
		}
	}

	//Taskset is unschedulable. Exit. Nothing to be done.
	if(m_used > num_CPUs) 
	{
		fprintf(stderr, "Error: System unschedulable. Exiting.");
		killpg(process_group, SIGKILL);
		return;
	}

	//Only schedulable using minimum.
	else if (m_used == num_CPUs)
	{
		goto give_cpus;
	}

	//Schedulable. Now allocate CPUs.
	//Begin by determining benefit of assigning next processor to each task.
	//Place tasks in heap based on potential benefit, z. 
	sched_heap.clear();
	for(int i=0; i<schedule.count(); i++)
	{
		
	
		//We're starting over with CPU allocation.
		(schedule.get_task(i))->set_current_lowest_CPU(-1);	
	
		if(((schedule.get_task(i))->get_current_CPUs() < (schedule.get_task(i))->get_max_CPUs()) && (schedule.get_task(i))->get_changeable())
		{
			double x = 1.0/(schedule.get_task(i))->get_elasticity()*(std::pow((schedule.get_task(i))->get_max_utilization()-(schedule.get_task(i))->get_current_utilization(),2));
			(schedule.get_task(i))->set_current_CPUs((schedule.get_task(i))->get_current_CPUs()+1);
			double y = 1.0/(schedule.get_task(i))->get_elasticity()*(std::pow((schedule.get_task(i))->get_max_utilization()-(schedule.get_task(i))->get_current_utilization(),2));
			double z = x - y;
			(schedule.get_task(i))->set_current_CPUs((schedule.get_task(i))->get_current_CPUs()-1);

			//Create a new node for heap. i is index. Weight by z.
			struct sched_pair node(i,z);

			//Add node to the heap.
			sched_heap.push_back(node);
		}
	}
	
	//Turn vector into a heap.
	std::make_heap(sched_heap.begin(), sched_heap.end());		

	while(sched_heap.size()>0 && m_used < num_CPUs)
	{
		//Removes from "heap" but actually moves the item to the back of the data structure.
		std::pop_heap (sched_heap.begin(),sched_heap.end()); 
			
		//Find the actual max
		int max_index = sched_heap.back().index;
		
		//Assign processor to this task
		(schedule.get_task(max_index))->set_current_CPUs((schedule.get_task(max_index))->get_current_CPUs()+1);
		m_used+=1;
	
		//std::cout << m_used << " " << num_CPUs << " " << (schedule.get_task(max_index))->get_current_CPUs() << " " << (schedule.get_task(max_index))->get_max_CPUs() << std::endl;
	
		//See if we can assign this task more processors
		if(m_used < num_CPUs && (schedule.get_task(max_index))->get_current_CPUs() < (schedule.get_task(max_index))->get_max_CPUs()){
			double x = 1.0/(schedule.get_task(max_index))->get_elasticity()*(std::pow((schedule.get_task(max_index))->get_max_utilization()-(schedule.get_task(max_index))->get_current_utilization(),2));
			(schedule.get_task(max_index))->set_current_CPUs((schedule.get_task(max_index))->get_current_CPUs()+1);
                        double y = 1.0/(schedule.get_task(max_index))->get_elasticity()*(std::pow((schedule.get_task(max_index))->get_max_utilization()-(schedule.get_task(max_index))->get_current_utilization(),2));
                        double z = x - y;
                        (schedule.get_task(max_index))->set_current_CPUs((schedule.get_task(max_index))->get_current_CPUs()-1);

                        //Update weight and reinsert task into heap.
                        sched_heap.back().weight = z;
			std::push_heap(sched_heap.begin(), sched_heap.end());

		}
		else {
			//It can't get any more CPUs. Actually remove task from the data_structure;		
			sched_heap.pop_back();
		}
	}
*/
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
				fprintf(stderr, "Error in task %d: all tasks should have had lowest CPU cleared.\n",i);
				killpg(process_group, SIGKILL);
                		return;
			}

			(schedule.get_task(i))->set_current_lowest_CPU(next_CPU);
			next_CPU += (schedule.get_task(i))->get_current_CPUs();

			if(next_CPU > num_CPUs+1)
			{
				fprintf(stderr, "Error in task %d: too many CPUs have been allocated.%d %d\n",i, next_CPU, num_CPUs);
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
			//fprintf(stderr,"Down in bottom part. Task %d has %d current CPUs and %d previous CPUs\n",i,(schedule.get_task(i))->get_current_CPUs(),(schedule.get_task(i))->get_previous_CPUs());

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
					//fprintf(stderr,"FIRST CASE %d %d\n",(schedule.get_task(i))->get_CPUs_gained(), (schedule.get_task(j))->get_CPUs_gained());

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
					//
					//fprintf(stderr,"SECOND CASE %d %d\n",(schedule.get_task(i))->get_CPUs_gained(), (schedule.get_task(j))->get_CPUs_gained());

					int difference = abs((schedule.get_task(j))->get_CPUs_gained()) - abs((schedule.get_task(i))->get_CPUs_gained());
                                        if(difference >= 0)
                                        {
                                                int amount = abs((schedule.get_task(i))->get_CPUs_gained());
                                                (schedule.get_task(j))->set_CPUs_gained((schedule.get_task(j))->get_CPUs_gained() - amount);
                                                (schedule.get_task(i))->set_CPUs_gained((schedule.get_task(i))->get_CPUs_gained() + amount);
                                                (schedule.get_task(j))->update_give(i,-1*amount);
                                                (schedule.get_task(i))->update_give(j,amount);
                                        }
					else
                                        {
                                                int amount = abs((schedule.get_task(j))->get_CPUs_gained());
                                                (schedule.get_task(j))->set_CPUs_gained((schedule.get_task(j))->get_CPUs_gained() - amount);
                                                (schedule.get_task(i))->set_CPUs_gained((schedule.get_task(i))->get_CPUs_gained() + amount);
                                                (schedule.get_task(j))->update_give(i,-1*amount);
                                                (schedule.get_task(i))->update_give(j,amount);
                                        }
				}
				//else
				//{fprintf(stderr,"THIRD CASE %d->%d  %d-> %d\n",i,(schedule.get_task(i))->get_CPUs_gained(), j,(schedule.get_task(j))->get_CPUs_gained());}
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
					//fprintf(stderr,"NOW FIRST CASE %d gives %d : %d CPUS\n",i, j, amount_gives);
				
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
									fprintf(stderr,"Task %d should be sending CPU %d to task %d.\n",i,k,j);
									(schedule.get_task(i))->set_transfer(j,k,true);
									//(schedule.get_task(i))->clr_active(k); //James add 7/8/18
									//(schedule.get_task(i))->set_passive(k); //James add 7/8/18
									(schedule.get_task(j))->set_receive(i,k,true);
									//(schedule.get_task(j))->clr_passive(k); //James add 7/8/18
                                                                        //(schedule.get_task(j))->set_active(k); //James add 7/8/18
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
                                                                        fprintf(stderr,"Task %d should be sending CPU %d to task %d.\n",i,k,j);
									(schedule.get_task(i))->set_transfer(j,k,true);
									//(schedule.get_task(i))->clr_active(k); //James add 7/8/18
                                                                        //(schedule.get_task(i))->set_passive(k); //James add 7/8/18
									(schedule.get_task(j))->set_receive(i,k,true);
									//(schedule.get_task(j))->clr_passive(k); //James add 7/8/18
                                                                        //(schedule.get_task(j))->set_active(k); //James add 7/8/18
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
					//fprintf(stderr,"NOW SECOND CASE %d gives %d : %d CPUS\n",j, i, amount_gives);
						
					cpu_set_t first;
					cpu_set_t second;			
					CPU_ZERO(&first);
					CPU_ZERO(&second);
					CPU_ZERO(&overlap);
                                        for(int t=1; t<=NUMCPUS; t++){
                                                /*if(schedule.get_task(j)->get_active(t) && schedule.get_task(i)->get_passive(t))
                                                {
                                                        CPU_SET(t,&overlap);
                                                }*/
						if(schedule.get_task(j)->get_active(t))
						{
							CPU_SET(t, &first);

						}
							//fprintf(stderr,"%d\n",CPU_COUNT(&first));
						if(schedule.get_task(i)->get_passive(t))
						{
							CPU_SET(t, &second);
				
						}
							//fprintf(stderr,"%d\n",CPU_COUNT(&second));
						

                                        }
	
					CPU_AND(&overlap, &first, &second);

					int amount_overlap = CPU_COUNT(&overlap);

					/*fprintf(stderr,"ACTIVE:");
					for(int t=1; t<=NUMCPUS; t++)
					{
						if(schedule.get_task(j)->get_active(t))
						fprintf(stderr,"%d ",t);
					}
					fprintf(stderr,"\nPASSIVE:");
                                        for(int t=1; t<=NUMCPUS; t++)
                                        {
                                                if(schedule.get_task(i)->get_passive(t))
                                                fprintf(stderr,"%d ",t);
                                        }
					fprintf(stderr,"\nOVERLAP:");
                                        for(int t=1; t<=NUMCPUS; t++)
                                        {
                                                if(CPU_ISSET(t,&overlap))
                                                fprintf(stderr,"%d ",t);
                                        }*/


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
									//(schedule.get_task(j))->clr_active(k); //James add 7/8/18
                                                                        //(schedule.get_task(j))->set_passive(k); //James add 7/8/18
									(schedule.get_task(i))->set_receive(j,k,true);
									//(schedule.get_task(i))->clr_passive(k); //James add 7/8/18
                                                                        //(schedule.get_task(i))->set_active(k); //James add 7/8/18
									fprintf(stderr,"Task %d should be sending CPU %d to task %d.\n",j,k,i);
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
									//(schedule.get_task(j))->clr_active(k); //James add 7/8/18
                                                                        //(schedule.get_task(j))->set_passive(k); //James add 7/8/18
									(schedule.get_task(i))->set_receive(j,k,true);
									//(schedule.get_task(i))->clr_passive(k); //James add 7/8/18
                                                                        //(schedule.get_task(i))->set_active(k); //James add 7/8/18
									fprintf(stderr,"Task %d should be sending CPU %d to task %d.\n",j,k,i);
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
