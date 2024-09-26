#include "scheduler.h"
#include "taskData.h"
#include "print_module.h"

#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cerrno>
#include <float.h>
#include <map>
#include <tuple>

class Schedule * Scheduler::get_schedule(){
	return &schedule;
}

TaskData * Scheduler::add_task(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_){
	
	//add the task to the legacy schedule object, but also add to vector
	//to make the scheduler much easier to read and work with.
	auto taskData_object = schedule.add_task(elasticity_, num_modes_, work_, span_, period_);

	task_table.push_back(std::vector<task_mode>());

	for (int j = 0; j < num_modes_; j++){

		task_mode item;
		item.cpuLoss = (1.0 / taskData_object->get_elasticity() * (std::pow(taskData_object->get_max_utilization() - (taskData_object->get_work(j) / taskData_object->get_period(j)), 2)));
		item.gpuLoss = 0;
		item.cores = taskData_object->get_CPUs(j);
		item.sms = 0;

		task_table.at(task_table.size() - 1).push_back(item);

	}

	return taskData_object;
}

//static vector size
std::vector<std::vector<Scheduler::task_mode>> Scheduler::task_table(100, std::vector<task_mode>(100));

//Implement scheduling algorithm
void Scheduler::do_schedule(size_t maxCPU, size_t maxSMS){

	//dynamic programming table
	int N = task_table.size();
    std::vector<std::vector<std::vector<std::pair<double, double>>>> dp(N + 1, std::vector<std::vector<std::pair<double, double>>>(maxCPU + 1, std::vector<std::pair<double, double>>(maxSMS + 1, {100000, 100000})));
    std::map<std::tuple<int,int,int>, std::vector<int>> solutions;

	//First time through Make sure we have enough CPUs 
	//in the system and determine practical max for each task.	
	if(first_time) {

		int min_required = 0;

		//Determine minimum required processors
		for(int i=0; i<schedule.count(); i++){

			min_required += (schedule.get_task(i))->get_min_CPUs();
			(schedule.get_task(i))->set_CPUs_gained(0);

		}

		//Determine the practical maximum. This is how many are left after each task has been given its minimum.
		for(int i=0; i<schedule.count(); i++){

			if((NUMCPUS - min_required + (schedule.get_task(i))->get_min_CPUs()) < (schedule.get_task(i))->get_max_CPUs())
				(schedule.get_task(i))->set_practical_max_CPUs( NUMCPUS - min_required + (schedule.get_task(i))->get_min_CPUs());

			else
				(schedule.get_task(i))->set_practical_max_CPUs((schedule.get_task(i))->get_max_CPUs());

		}
	}

	//Execute double knapsack algorithm
    for (size_t i = 1; i <= num_tasks; i++) {

        for (size_t w = 0; w <= maxCPU; w++) {

            for (size_t v = 0; v <= maxSMS; v++) {

                //invalid state
                dp[i][w][v] = {-1, -1};  
                
				//if the class we are considering is not allowed to switch modes
				//just treat it as though we did check it normally, and only allow
				//looking at the current mode.
				if(!(schedule.get_task(i - 1))->get_changeable()){

						auto item = task_table.at(i - 1).at((schedule.get_task(i - 1))->get_current_mode());

						//if item fits in both sacks
						if ((w >= item.cores) && (v >= item.sms) && (dp[i - 1][w - item.cores][v - item.sms].first != -1)) {

							int newCPULoss = dp[i - 1][w - item.cores][v - item.sms].first - item.cpuLoss;
							int newGPULoss = dp[i - 1][w - item.cores][v - item.sms].second - item.gpuLoss;
							
							//if found solution is better, update
							if ((newCPULoss + newGPULoss) > (dp[i][w][v].first + dp[i][w][v].second)) {

								dp[i][w][v] = {newCPULoss, newGPULoss};

								solutions[{i, w, v}] = solutions[{i - 1, w - item.cores, v - item.sms}];
								solutions[{i, w, v}].push_back((schedule.get_task(i - 1))->get_current_mode());

							}
						}
					}

				else{

					//for each item in class
					for (size_t j = 0; j < task_table.at(i - 1).size(); j++) {

						auto item = task_table.at(i - 1).at(j);

						//if item fits in both sacks
						if ((w >= item.cores) && (v >= item.sms) && (dp[i - 1][w - item.cores][v - item.sms].first != -1)) {

							int newCPULoss = dp[i - 1][w - item.cores][v - item.sms].first - item.cpuLoss;
							int newGPULoss = dp[i - 1][w - item.cores][v - item.sms].second - item.gpuLoss;
							
							//if found solution is better, update
							if ((newCPULoss + newGPULoss) > (dp[i][w][v].first + dp[i][w][v].second)) {

								dp[i][w][v] = {newCPULoss, newGPULoss};

								solutions[{i, w, v}] = solutions[{i - 1, w - item.cores, v - item.sms}];
								solutions[{i, w, v}].push_back(j);

							}
						}
					}
				}
            }
        }
    }

    //return optimal solution
	auto result = solutions[{N, maxCPU, maxSMS}];
	solutions.clear();

	//update the tasks
	std::string mode_strings = "";
	for (size_t i = 0; i < result.size(); i++)
		mode_strings += "Task " + std::to_string(i) + " is now in mode: " + std::to_string(result.at(i)) + "\n";
	print_module::print(std::cout, "========================= \nNew Schedule Layout:\n", mode_strings, "Total Loss from Mode Change: ", 100000 - dp[N][maxCPU][maxSMS].first, "\n=========================\n");

	for (int i=0; i<schedule.count(); i++)
		(schedule.get_task(i))->set_current_mode(result.at(i), false);

	//greedily give cpus on first run
	if(first_time) {

		int next_CPU = 1;

		//Actually assign CPUs to tasks. Start with 1.
		for(int i = 0; i < schedule.count(); i++){

			if((schedule.get_task(i))->get_current_lowest_CPU() > 0){

				print_module::print(std::cerr, "Error in task ", i, ": all tasks should have had lowest CPU cleared. (this likely means memory was not cleaned up)\n");
				killpg(process_group, SIGKILL);
                return;

			}

			(schedule.get_task(i))->set_current_lowest_CPU(next_CPU);
			next_CPU += (schedule.get_task(i))->get_current_CPUs();

			if(next_CPU > num_CPUs + 1){

				print_module::print(std::cerr, "Error in task ", i, ": too many CPUs have been allocated.", next_CPU, " ", num_CPUs, " \n");
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
				
					for(int t=1; t<=NUMCPUS; t++)
					{
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
									print_module::print(std::cerr, "Task ", i, " should be sending CPU ", k, " to task ", j, ".\n");
									
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
									print_module::print(std::cerr, "Task ", i, " should be sending CPU ", k, " to task ", j, ".\n");

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
									
									print_module::print(std::cerr, "Task ", j, "  should be sending CPU ", k, " to task ", i, ".\n");

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
									
									print_module::print(std::cerr, "Task ", j, "  should be sending CPU ", k, " to task ", i, ".\n");

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

void Scheduler::setTermination(){
	schedule.setTermination();
}