#include "taskData.h"	

TaskData::TaskData(){}

TaskData::TaskData(bool free_resources){

	//open the 3 message queues which facilitate the process of giving CPUs or GPUs to other tasks

	//queue 1 is used to signal that a task is giving up CPUs
	if ((queue_one = msgget(98173, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 1.\n");
		kill(0, SIGTERM);

	}

	//queue 2 is used to signal that a task is giving CPUs to another task
	if ((queue_two = msgget(98174, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 2.\n");
		kill(0, SIGTERM);

	}

	//queue 3 is used to signal that a task should give up the resources contained in the mask to the task specified
	if ((queue_three = msgget(98175, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 3.\n");
		kill(0, SIGTERM);

	}

	index = counter++;

}

static std::vector<std::tuple<int,int,int,int>> computeModeResources(double CpA, double L_A,
   										                    double CpB, double L_B,
        									                double T,
															double CpC, double L_C,
															double CpD, double L_D,
															std::tuple<int, float> equivalent_vector[4]){

	// We want to use A as the base processor
	// we assume B, C, and D can all be equivalent to A
	// if this is the case, then we need to consider the
	// canonical form of the DAG which is the DAG with the span
	// completed as a dependency of all other work in the DAG.
	// From this, we can use the simple equation:
	//
	//       T = L_A + (CpA - L_A) / (mA * (average_processor_speed))
	//
	// where average_processor_speed is the average speed of the processors
	// assigned to the DAG. Once we generate all possible combinations of
	// processors which can be assigned to the DAG, we can then use the
	// method we used before to check for all the ways that the other processors
	// can be assigned when we divide the period
	std::vector<std::pair<double, double>> processor_information = { {CpA, L_A}, {CpB, L_B}, {CpC, L_C}, {CpD, L_D} };
	std::vector<int> system_processors = {NUM_PROCESSOR_A, NUM_PROCESSOR_B, NUM_PROCESSOR_C, NUM_PROCESSOR_D};

	//we also remove the time for the core handoff latency
	T -= MAX_CORE_HANDOFF_LATENCY_NSEC;

	//Im tired of this. Just generate all possible combinations and we are going 
	//to walk through them all. If there is a way to meet the deadline, we keep the 
	//combination
	std::vector<std::tuple<int,int,int,int>> valid_combinations;

	for (int a = 1; a < 128; a++){

		if (a > NUM_PROCESSOR_A)
			continue;

		//if the count is more than 0 and we dont actually have any work to 
		//do on this processor, we can just skip it
		if (a > 0 && CpA == 0 && std::get<0>(equivalent_vector[0]) == 0)
			continue;

		for (int b = 0; b < 128; b++){

			if (b > NUM_PROCESSOR_B)
				continue;

			//if the count is more than 0 and we dont actually have any work to 
			//do on this processor, we can just skip it
			if (b > 0 && CpB == 0 && std::get<0>(equivalent_vector[1]) == 1)
				continue;

			for (int c = 0; c < 128; c++){

				if (c > NUM_PROCESSOR_C)
					continue;

				//if the count is more than 0 and we dont actually have any work to 
				//do on this processor, we can just skip it
				if (c > 0 && CpC == 0 && std::get<0>(equivalent_vector[2]) == 2)
					continue;

				for (int d = 0; d < 128; d++){

					if (d > NUM_PROCESSOR_D)
						continue;

					//if the count is more than 0 and we dont actually have any work to 
					//do on this processor, we can just skip it
					if (d > 0 && CpD == 0 && std::get<0>(equivalent_vector[3]) == 3)
						continue;

					//first we calculate how much time is spent in the A
					//portion of the DAG
					double processor_average = 0;
					int total_processors_used = 0;

					double time_left = T;

					for (int i = 0; i < 4; i++){

						if (std::get<0>(equivalent_vector[i]) == 0){
							
							switch (i){

								case 0:
									processor_average += std::get<1>(equivalent_vector[i]) * a;
									total_processors_used += a;
									break;
								case 1:
									processor_average += std::get<1>(equivalent_vector[i]) * b;
									total_processors_used += b;
									break;
								case 2:
									processor_average += std::get<1>(equivalent_vector[i]) * c;
									total_processors_used += c;
									break;
								case 3:
									processor_average += std::get<1>(equivalent_vector[i]) * d;
									total_processors_used += d;
									break;

							}

						}

					}

					if ((total_processors_used == 0 ) && (CpA > 0))
						continue;
					
					//we now have all the processors which are equivalent to A
					//and can contribute to the first part of the DAG. Calculate
					//how long it takes to complete the A portion of the DAG
					double time_to_complete_A = (total_processors_used > 0 ) ? (L_A + ((CpA - L_A) / processor_average)) : 0;

					//if it's longer than the period, then we can skip ahead
					if (time_to_complete_A > time_left)
						continue;

					//now we can calculate how much time is left for the B portion of the DAG
				 	time_left -= time_to_complete_A;

					//now we need to determine which processors are working on the 
					//B portion of the DAG
					if (std::get<0>(equivalent_vector[1]) == 1){

						processor_average = 0;
						total_processors_used = 0;

						for (int i = 1; i < 4; i++){

							if (std::get<0>(equivalent_vector[i]) == 1){
								
								switch (i){

									case 0:
										processor_average += std::get<1>(equivalent_vector[i]) * a;
										total_processors_used += a;
										break;
									case 1:
										processor_average += std::get<1>(equivalent_vector[i]) * b;
										total_processors_used += b;
										break;
									case 2:
										processor_average += std::get<1>(equivalent_vector[i]) * c;
										total_processors_used += c;
										break;
									case 3:
										processor_average += std::get<1>(equivalent_vector[i]) * d;
										total_processors_used += d;
										break;

								}

							}

						}

						if ((total_processors_used == 0 ) && (CpB > 0))
							continue;

						//we now have all the processors which are equivalent to B
						//and can contribute to the first part of the DAG. Calculate
						//how long it takes to complete the B portion of the DAG
						double time_to_complete_B = (total_processors_used > 0 ) ? L_B + ((CpB - L_B) / ((processor_average / total_processors_used) * (total_processors_used))) : 0;

						//if it's longer than the period, then we can skip ahead
						if (time_to_complete_B > time_left)
							continue;

						time_left -= time_to_complete_B;

					}

					//now we need to determine which processors are working on the
					//C portion of the DAG
					if (std::get<0>(equivalent_vector[2]) == 2){

						processor_average = 0;
						total_processors_used = 0;

						for (int i = 2; i < 4; i++){

							if (std::get<0>(equivalent_vector[i]) == 2){
								
								processor_average += std::get<1>(equivalent_vector[i]);
								
								switch (i){

									case 0:
										processor_average += std::get<1>(equivalent_vector[i]) * a;
										total_processors_used += a;
										break;
									case 1:
										processor_average += std::get<1>(equivalent_vector[i]) * b;
										total_processors_used += b;
										break;
									case 2:
										processor_average += std::get<1>(equivalent_vector[i]) * c;
										total_processors_used += c;
										break;
									case 3:
										processor_average += std::get<1>(equivalent_vector[i]) * d;
										total_processors_used += d;
										break;

								}

							}

						}

						if ((total_processors_used == 0 ) && (CpC > 0))
							continue;

						//we now have all the processors which are equivalent to C
						//and can contribute to the first part of the DAG. Calculate
						//how long it takes to complete the C portion of the DAG
						double time_to_complete_C = (total_processors_used > 0 ) ? L_C + ((CpC - L_C) / ((processor_average / total_processors_used) * (total_processors_used))) : 0;

						//if it's longer than the period, then we can skip ahead
						if (time_to_complete_C > time_left)
							continue;

						time_left -= time_to_complete_C;

					}

					//now we need to determine which processors are working on the
					//D portion of the DAG
					if (std::get<0>(equivalent_vector[3]) == 3){

						processor_average = 0;
						total_processors_used = 0;

						for (int i = 3; i < 4; i++){

							if (std::get<0>(equivalent_vector[i]) == 3){
								
								processor_average += std::get<1>(equivalent_vector[i]);
								
								switch (i){

									case 0:
										processor_average += std::get<1>(equivalent_vector[i]) * a;
										total_processors_used += a;
										break;
									case 1:
										processor_average += std::get<1>(equivalent_vector[i]) * b;
										total_processors_used += b;
										break;
									case 2:
										processor_average += std::get<1>(equivalent_vector[i]) * c;
										total_processors_used += c;
										break;
									case 3:
										processor_average += std::get<1>(equivalent_vector[i]) * d;
										total_processors_used += d;
										break;
								}

							}

						}

						if ((total_processors_used == 0 ) && (CpD > 0)){

						std::cout << "FOR WORK D : " << CpD << " I HAVE NO PROCESSORS WHEN I LOOKED AT " << a << " A and " << b << " B and " << c << " C and " << d << " D\n";

						continue;

					}
						//we now have all the processors which are equivalent to D
						//and can contribute to the first part of the DAG. Calculate
						//how long it takes to complete the D portion of the DAGs
						double time_to_complete_D = (total_processors_used > 0 ) ? L_D + ((CpD - L_D) / ((processor_average / total_processors_used) * (total_processors_used))) : 0;

						//if it's longer than the period, then we can skip ahead
						if (time_to_complete_D > time_left){

						std::cout << "WITH TIME TO COMPLETE LEFT: " << time_left << " I CANNOT COMPLETE A D PORTION WHICH TAKES " << time_to_complete_D << " WHEN I LOOKED AT " << a << " A and " << b << " B and " << c << " C and " << d << " D\n";

						continue;

					}

						time_left -= time_to_complete_D;

					}

					//if we made it down here, then this is a valid combination 
					//which never exceeded the original budget of time T
					valid_combinations.push_back(std::make_tuple(a, b, c, d));
					//std::cout << "Possible Entry: " << a << " " << b << " " << c << " " << d << std::endl;

				}

			}

		}

	}

	//now we need to filter out the modes which are not the minimum number of processors needed for the A and B portions of the DAG
	std::vector<std::tuple<int,int,int,int>> result_filtered;

	//a bool vector to keep track of dead entries
	std::vector<bool> dead_entries(valid_combinations.size(), false);

	//FIXME: COND_VEC NEEDS TO BE PASSED IN
	int cond_vec[4] = {1,1,0,0};

	//remove all combinations (mark as dead) which do not meet
	//the minimum held processors requirement
	for (size_t i = 0; i < valid_combinations.size(); i++){

		//filter non-held minimums
		bool holds_minimum_proc = false;

		int processor_counts[4] = {std::get<0>(valid_combinations[i]), std::get<1>(valid_combinations[i]), std::get<2>(valid_combinations[i]), std::get<3>(valid_combinations[i])};
		
		for (int j = 0; j < 4; j++)
			if (cond_vec[j] == 1 && processor_counts[j] >= 1)
				holds_minimum_proc = true;
		
		if (!holds_minimum_proc)
			dead_entries[i] = true;
		
	}

	//filter out duplicates which just have higher computational loads
	//than others. Starting with the last entry, we select one combination
	//and we extract the "prefix" of the combination (all other processors
	//needed). If we make it through the entire list without finding 
	//another entry that has the same prefix, but with a lower processor
	//that we are looking at value (value at i), then we add it to the 
	//keeping list. Otherwise, we keep the better one and continue.
	for (int i = 3; i >= 0; i--){

		for (size_t combinations = 0; combinations < valid_combinations.size(); combinations++){

			if (dead_entries[combinations])
				continue;

			auto currently_selected = valid_combinations[combinations];

			int prefix[4] = {std::get<0>(currently_selected), std::get<1>(currently_selected), std::get<2>(currently_selected), std::get<3>(currently_selected)};

			for (size_t other_combinations = combinations; other_combinations < valid_combinations.size(); other_combinations++){

				if (dead_entries[other_combinations] || other_combinations == combinations)
					continue;

				int current_prefix[4] = {std::get<0>(valid_combinations[other_combinations]), std::get<1>(valid_combinations[other_combinations]), std::get<2>(valid_combinations[other_combinations]), std::get<3>(valid_combinations[other_combinations])};

				bool all_other_processors_equal = true;

				for (int j = 0; j < 4; j++){

					if (j == i)
						continue;

					if (prefix[j] != current_prefix[j]){

						all_other_processors_equal = false;
						break;

					}

				}

				if (all_other_processors_equal){

					if (prefix[i] > current_prefix[i]){

						//we have found a better candidate, so this
						//is the new one to keep
						for (int k = 0; k < 4; k++)
							prefix[k] = current_prefix[k];

						//update currently selected	
						currently_selected = valid_combinations[other_combinations];

						dead_entries[combinations] = true;

					}

					else {

						//the current one is better, so we mark the other one as dead
						dead_entries[other_combinations] = true;

					}

				}

			}

		}

	}

	//now we can make the final list of valid combinations
	for (size_t i = 0; i < valid_combinations.size(); i++){

		if (!dead_entries[i]){
			result_filtered.push_back(valid_combinations[i]);
		}

	}

    return result_filtered;
    
}

TaskData::TaskData(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, 
														timespec * processor_B_work_, timespec * processor_B_span_, timespec * processor_B_period_, 
														timespec * processor_C_work_, timespec * processor_C_span_, timespec * processor_C_period_, 
														timespec * processor_D_work_, timespec * processor_D_span_, timespec * processor_D_period_, bool safe, std::tuple<int, float> equivalent_vector[4], bool print) : 	
																													
																													index(counter++), changeable(true), 
																													can_reschedule(false), num_adaptations(0),  
																													elasticity(elasticity_), num_modes(num_modes_), 
																													max_utilization(0), max_processors_A(0), min_processors_A(NUM_PROCESSOR_A), 
																													max_processors_B(0), min_processors_B(NUM_PROCESSOR_A), 
																													max_processors_C(0), min_processors_C(NUM_PROCESSOR_C),
																													max_processors_D(0), min_processors_D(NUM_PROCESSOR_D),
																													processors_A_gained(0), processors_B_gained(0),
																													processors_C_gained(0), processors_D_gained(0),
																													practical_max_processors_A(max_processors_A), current_lowest_processor_A(-1), 
																													practical_max_processors_B(max_processors_B), current_lowest_processor_B(0),
																													practical_max_processors_C(max_processors_C), current_lowest_processor_C(-1),
																													practical_max_processors_D(max_processors_D), current_lowest_processor_D(0),
																													percentage_workload(1.0), current_period({0,0}), 
																													current_work({0,0}), current_span({0,0}), 
																													current_utilization(0.0), current_processors_A(0), previous_processors_A(0), 
																													current_processors_B(0), previous_processors_B(0),
																													current_processors_C(0), previous_processors_C(0),
																													current_processors_D(0), previous_processors_D(0),
																													permanent_processor_A(-1), max_work({0,0}), current_virtual_mode(0){
	
	if (num_modes > MAXMODES){

		print_module::print(std::cerr, "ERROR: No task can have more than ", MAXMODES,  " modes.\n");
		kill(0, SIGTERM);
	
	}		

	int mode_options = 0;
	int next_position = 0;

	//print the view the scheduler has of the topology 
	//of the chips in the system
	if (print){

		// Print processor configuration for testing
		print_module::print(std::cerr, "TaskData view of Processor Configuration:\n");
		print_module::print(std::cerr, "  A: type=", std::get<0>(equivalent_vector[0]), " (", 
			(std::get<0>(equivalent_vector[0]) == 0 ? "A" : std::get<0>(equivalent_vector[0]) == 1 ? "B" : 
			std::get<0>(equivalent_vector[0]) == 2 ? "C" : "D"), "), ratio=", std::get<1>(equivalent_vector[0]), "\n");
		print_module::print(std::cerr, "  B: type=", std::get<0>(equivalent_vector[1]), " (", 
			(std::get<0>(equivalent_vector[1]) == 0 ? "A" : std::get<0>(equivalent_vector[1]) == 1 ? "B" : 
			std::get<0>(equivalent_vector[1]) == 2 ? "C" : "D"), "), ratio=", std::get<1>(equivalent_vector[1]), "\n");
		print_module::print(std::cerr, "  C: type=", std::get<0>(equivalent_vector[2]), " (", 
			(std::get<0>(equivalent_vector[2]) == 0 ? "A" : std::get<0>(equivalent_vector[2]) == 1 ? "B" : 
			std::get<0>(equivalent_vector[2]) == 2 ? "C" : "D"), "), ratio=", std::get<1>(equivalent_vector[2]), "\n");
		print_module::print(std::cerr, "  D: type=", std::get<0>(equivalent_vector[3]), " (", 
			(std::get<0>(equivalent_vector[3]) == 0 ? "A" : std::get<0>(equivalent_vector[3]) == 1 ? "B" : 
			std::get<0>(equivalent_vector[3]) == 2 ? "C" : "D"), "), ratio=", std::get<1>(equivalent_vector[3]), "\n");

	}

	//determine resources
	for (int i = 0; i < num_modes; i++){

		//fetch the mode parameters
		timespec work_l = work_[i];
		timespec span_l = span_[i];
		timespec period_l = period_[i];
		timespec processor_B_work_l = processor_B_work_[i];
		timespec processor_B_span_l = processor_B_span_[i];
		timespec processor_B_period_l = processor_B_period_[i];
		timespec processor_C_work_l = processor_C_work_[i];
		timespec processor_C_span_l = processor_C_span_[i];
		timespec processor_C_period_l = processor_C_period_[i];
		timespec processor_D_work_l = processor_D_work_[i];
		timespec processor_D_span_l = processor_D_span_[i];
		timespec processor_D_period_l = processor_D_period_[i];

		//fetch each mode parameter as a long
		double work_long = get_timespec_in_ns(work_l);
		double span_long = get_timespec_in_ns(span_l);
		double period_long = get_timespec_in_ns(period_l);
		double processor_B_work_long = get_timespec_in_ns(processor_B_work_l);
		double processor_B_span_long = get_timespec_in_ns(processor_B_span_l);
		double processor_C_work_long = get_timespec_in_ns(processor_C_work_l);
		double processor_C_span_long = get_timespec_in_ns(processor_C_span_l);
		double processor_D_work_long = get_timespec_in_ns(processor_D_work_l);
		double processor_D_span_long = get_timespec_in_ns(processor_D_span_l);

		//set the equivalent processors (only if the equivalent processor is 0)
		processors_equivalent_to_A[0] = std::get<0>(equivalent_vector[0]) == 0;
		processors_equivalent_to_A[1] = std::get<0>(equivalent_vector[1]) == 0;
		processors_equivalent_to_A[2] = std::get<0>(equivalent_vector[2]) == 0;
		processors_equivalent_to_A[3] = std::get<0>(equivalent_vector[3]) == 0;

		//pass to the computeModeResources function
		auto resources = computeModeResources(work_long, span_long, processor_B_work_long, processor_B_span_long, period_long, processor_C_work_long, processor_C_span_long, processor_D_work_long, processor_D_span_long, equivalent_vector);
		mode_options += resources.size();

		if (resources.size() == 0){
			print_module::print(std::cerr, "ERROR: No resources found for task ", index, " in mode ", i, "\n");
		}

		//loop over the resources and store them in the table
		for (auto res : resources){

			//store the processor A and processor B resources
			//FIXME: REPLAC EWITH REAL C AND D
			processors_A[next_position] = std::get<0>(res);
			processors_B[next_position] = std::get<1>(res);
			processors_C[next_position] = std::get<2>(res);
			processors_D[next_position] = std::get<3>(res);

			//map the mode
			mode_map[next_position] = i;

			//update the parameters
			if (processors_A[next_position] > max_processors_A)
				max_processors_A = processors_A[next_position];

			if (processors_A[next_position] < min_processors_A)
				min_processors_A = processors_A[next_position];

			if (processors_B[next_position] > max_processors_B)
				max_processors_B = processors_B[next_position];

			if (processors_B[next_position] < min_processors_B)
				min_processors_B = processors_B[next_position];

			if (processors_C[next_position] > max_processors_C)
				max_processors_C = processors_C[next_position];

			if (processors_C[next_position] < min_processors_C)
				min_processors_C = processors_C[next_position];

			if (processors_D[next_position] > max_processors_D)
				max_processors_D = processors_D[next_position];

			if (processors_D[next_position] < min_processors_D)
				min_processors_D = processors_D[next_position];

			//set the params for this mode
			work[next_position] = work_l;
			span[next_position] = span_l;
			period[next_position] = period_l;
			processor_B_work[next_position] = processor_B_work_l;
			processor_B_span[next_position] = processor_B_span_l;
			processor_B_period[next_position] = processor_B_period_l;
			processor_C_work[next_position] = processor_C_work_l;
			processor_C_span[next_position] = processor_C_span_l;
			processor_C_period[next_position] = processor_C_period_l;
			processor_D_work[next_position] = processor_D_work_l;
			processor_D_span[next_position] = processor_D_span_l;
			processor_D_period[next_position] = processor_D_period_l;

			//note which mode this came from
			owning_modes[next_position] = i;

			//update max utilization
			if (((work_l / period_l) + (processor_B_work_l / period_l) + (processor_C_work_l / period_l) + (processor_D_work_l / period_l)) > max_utilization)
				max_utilization = ((work_l / period_l) + (processor_B_work_l / period_l) + (processor_C_work_l / period_l) + (processor_D_work_l / period_l));

			if (++next_position >= MAXMODES){

				print_module::print(std::cerr, "ERROR: No task can have more than ", MAXMODES, " modes.\n");
				kill(0, SIGTERM);
			
			}

		}

		if (work[i] > max_work)
			max_work = work[i];

		if ((processor_B_work[i] != timespec({0, 0}) &&  processor_B_span[i] != timespec({0, 0})) ||
			(processor_C_work[i] != timespec({0, 0}) &&  processor_C_span[i] != timespec({0, 0})) ||
			(processor_D_work[i] != timespec({0, 0}) &&  processor_D_span[i] != timespec({0, 0})))
			is_pure_A_task = false;

	}

	print_module::print(std::cout, "Task ", index, " has ", num_modes, " modes:\n");
	for (int i = 0; i < num_modes; i++)
		print_module::print(std::cout, work_[i], " ", span_[i], " ", period_[i], " ", processor_B_work_[i], " ", processor_B_span_[i], " ", processor_C_work_[i], " ", processor_C_span_[i], " ", processor_C_period_[i], " ", processor_D_work_[i], " ", processor_D_span_[i], " ", processor_D_period_[i], "\n");
	print_module::print(std::cout, "\n");
	
	modes_originally_passed = num_modes;

	num_modes = mode_options;
	number_of_modes = mode_options;

	//check if we should be making this task safe from false negatives
	if (safe){

		//look through the lowest mode and for each possible state 
		//select a mode which represents the lowest possible state
		//and look through all the other mode_options, and count the
		//number of mode options which are stricly larger
		//than the current mode option. Ensure at least one candidate mode is present
		//from each mode transition possible

		int best_base_mode = 0;
		int best_base_mode_seen = 0;

		for (int i = 0; i < num_modes; i++){

			int current_processor_A = processors_A[i];
			int current_processor_B = processors_B[i];

			int distinct_modes_seen = 0;
			int seen_modes_utilizations[modes_originally_passed] = {0};

			//set the first seen mode as itself
			seen_modes_utilizations[owning_modes[i]] = 1;

			for (int j = 0; j < num_modes; j++){

				if (i != j){

					if (processors_A[j] >= current_processor_A && processors_B[j] >= current_processor_B){

						distinct_modes_seen++;
						seen_modes_utilizations[owning_modes[j]] += 1;

					}

				}

			}

			if (best_base_mode_seen < distinct_modes_seen){

				bool candidate_mode = true;
				for (int m = 0; m < modes_originally_passed; m++){

					if (seen_modes_utilizations[m] == 0)
						candidate_mode = false;

				}

				if (candidate_mode){

					best_base_mode = i;
					best_base_mode_seen = distinct_modes_seen;

				}

			}

		}


		//if we have a best base mode set, we can use that as
		//the standard by which we forcibly inflate all other 
		//modes
		if (best_base_mode_seen > 0){

			for (int i = 0; i < num_modes; i++){

				if (processors_A[i] < processors_A[best_base_mode])
					processors_A[i] = processors_A[best_base_mode];

				if (processors_B[i] < processors_B[best_base_mode])
					processors_B[i] = processors_B[best_base_mode];

				if (processors_C[i] < processors_C[best_base_mode])
					processors_C[i] = processors_C[best_base_mode];

				if (processors_D[i] < processors_D[best_base_mode])
					processors_D[i] = processors_D[best_base_mode];

			}

			print_module::print(std::cout, "Task ", index, " has been forced to inflate to mode ", best_base_mode, "\n");

		}

		else {

			print_module::print(std::cout, "Task ", index, " has no base mode to inflate to. The system cannot be safely scheduled.\n");
			exit(-1);

		}


	}

	//loop over all modes, and compare the allocated processor A
	//and processor B to all other modes in the task. If the task
	//has any mode where it gains processor A and loses processor B compared to 
	//any other mode in the task or vice versa, then the task is
	//combinatorially elastic
	for (int i = 0; i < num_modes; i++){

		for (int j = 0; j < num_modes; j++){

			if (i != j){

				if (processors_A[i] > processors_A[j] && processors_B[i] < processors_B[j]){

					combinatorially_elastic = true;
					break;

				}

				if (processors_A[i] < processors_A[j] && processors_B[i] > processors_B[j]){

					combinatorially_elastic = true;
					break;

				}

				if (processors_C[i] > processors_C[j] && processors_D[i] < processors_D[j]){

					combinatorially_elastic = true;
					break;

				}

				if (processors_C[i] < processors_C[j] && processors_D[i] > processors_D[j]){

					combinatorially_elastic = true;
					break;

				}

			}

		}

	}

	current_processors_A = min_processors_A;
	current_processors_B = min_processors_B;
	current_processors_C = min_processors_C;
	current_processors_D = min_processors_D;

	//clear the tables so I can actually read them when needed
	for (int i  = 0; i < MAXTASKS + 1; i++){

		for (int j = 0; j < NUM_PROCESSOR_B + 1; j++){

			processors_B_granted_from_other_tasks[i][j] = -1;

		}

		for (int j = 0; j < NUM_PROCESSOR_A + 1; j++){

			processors_A_granted_from_other_tasks[i][j] = -1;

		}

		for (int j = 0; j < NUM_PROCESSOR_C + 1; j++){

			processors_C_granted_from_other_tasks[i][j] = -1;

		}

		for (int j = 0; j < NUM_PROCESSOR_D + 1; j++){

			processors_D_granted_from_other_tasks[i][j] = -1;

		}

	}

	//open the 3 message queues which facilitate the process of giving CPUs or GPUs to other tasks

	//queue 1 is used to signal that a task is giving up CPUs
	if ((queue_one = msgget(98173, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 1.\n");
		kill(0, SIGTERM);

	}

	//queue 2 is used to signal that a task is giving CPUs to another task
	if ((queue_two = msgget(98174, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 2.\n");
		kill(0, SIGTERM);

	}

	//queue 3 is used to signal that a task should give up the resources contained in the mask to the task specified
	if ((queue_three = msgget(98175, 0666 | IPC_CREAT)) == -1){

		print_module::print(std::cerr, "Error: Failed to create message queue 3.\n");
		kill(0, SIGTERM);

	}
}

TaskData::~TaskData(){
}

int TaskData::counter = 0;	        

int TaskData::get_index(){
	return index;
}

int TaskData::get_number_of_modes(){
	return number_of_modes;
}

int TaskData::get_processors_A_gained(){
	return processors_A_gained;
}

void TaskData::set_processors_A_gained(int new_processors_A_gained){
	processors_A_gained = new_processors_A_gained;
}

int TaskData::get_processors_B_gained(){
	return processors_B_gained;
}

void TaskData::set_processors_B_gained(int new_processors_B_gained){
	processors_B_gained = new_processors_B_gained;
}

int TaskData::get_processors_C_gained(){
	return processors_C_gained;
}

void TaskData::set_processors_C_gained(int new_processors_C_gained){
	processors_C_gained = new_processors_C_gained;
}

int TaskData::get_processors_D_gained(){
	return processors_D_gained;
}

void TaskData::set_processors_D_gained(int new_processors_D_gained){
	processors_D_gained = new_processors_D_gained;
}

double TaskData::get_elasticity(){
	return elasticity;
}

double TaskData::get_max_utilization(){
	return max_utilization;
}

int TaskData::get_max_processors_A(){
	return max_processors_A;
}

int TaskData::get_min_processors_A(){
	return min_processors_A;
}

timespec TaskData::get_processor_B_work(int index){
	return processor_B_work[index];
}

timespec TaskData::get_processor_B_span(int index){
	return processor_B_span[index];
}

timespec TaskData::get_processor_B_period(int index){
	return processor_B_period[index];
}

int TaskData::get_processors_B(int index){
	return processors_B[index];
}

//processor C functions that can be compiled regardless of compiler
timespec TaskData::get_processor_C_work(int index){
	return processor_C_work[index];
}

timespec TaskData::get_processor_C_span(int index){
	return processor_C_span[index];
}

timespec TaskData::get_processor_C_period(int index){
	return processor_C_period[index];
}

int TaskData::get_processors_C(int index){
	return processors_C[index];
}

//processor D functions that can be compiled regardless of compiler
timespec TaskData::get_processor_D_work(int index){
	return processor_D_work[index];
}

timespec TaskData::get_processor_D_span(int index){
	return processor_D_span[index];
}

timespec TaskData::get_processor_D_period(int index){
	return processor_D_period[index];
}

int TaskData::get_processors_D(int index){
	return processors_D[index];
}

int TaskData::get_current_processors_B(){
	return current_processors_B;
}

int TaskData::get_current_processors_C(){
   return current_processors_C;
}

int TaskData::get_current_processors_D(){
	return current_processors_D;
}

int TaskData::get_max_processors_B(){
	return max_processors_B;
}

int TaskData::get_min_processors_B(){
	return min_processors_B;
}

int TaskData::get_max_processors_C(){
	return max_processors_C;
}

int TaskData::get_min_processors_C(){
	return min_processors_C;
}

int TaskData::get_max_processors_D(){
	return max_processors_D;
}

int TaskData::get_min_processors_D(){
	return min_processors_D;
}

timespec TaskData::get_current_span(){
	return current_span;
}

timespec TaskData::get_current_work(){
	return current_work;
}	

timespec TaskData::get_current_period(){
	return current_period;
}

int TaskData::get_current_processors_A(){
	return current_processors_A;
}

double TaskData::get_percentage_workload(){
	return percentage_workload;
}

bool TaskData::get_changeable(){
	return changeable;
}

int TaskData::get_current_lowest_processor_A(){
   return current_lowest_processor_A;
}

void TaskData::set_practical_max_processors_A(int new_value){
   practical_max_processors_A = new_value;
}

int TaskData::get_practical_max_processors_A(){
   return practical_max_processors_A;
}

void TaskData::set_current_lowest_processor_A(int _lowest){
   current_lowest_processor_A = _lowest;
}

void TaskData::set_practical_max_processors_B(int new_value){
   practical_max_processors_B = new_value;
}

int TaskData::get_practical_max_processors_B(){
   return practical_max_processors_B;
}

void TaskData::set_current_lowest_processor_B(int _lowest){
   current_lowest_processor_B = _lowest;
}

int TaskData::get_current_lowest_processor_B(){
   return current_lowest_processor_B;
}

void TaskData::set_practical_max_processors_C(int new_value){
   practical_max_processors_C = new_value;
}

int TaskData::get_practical_max_processors_C(){
   return practical_max_processors_C;
}

void TaskData::set_current_lowest_processor_C(int _lowest){
   current_lowest_processor_C = _lowest;
}

int TaskData::get_current_lowest_processor_C(){
   return current_lowest_processor_C;
}

void TaskData::set_practical_max_processors_D(int new_value){
   practical_max_processors_D = new_value;
}

int TaskData::get_practical_max_processors_D(){
   return practical_max_processors_D;
}

void TaskData::set_current_lowest_processor_D(int _lowest){
   current_lowest_processor_D = _lowest;
}

int TaskData::get_current_lowest_processor_D(){
   return current_lowest_processor_D;
}

int TaskData::get_real_mode(int mode){
	return mode_map[mode];
}

void TaskData::reset_mode_to_previous(){

	current_virtual_mode = previous_mode;
	current_work = work[current_virtual_mode];
	current_span = span[current_virtual_mode];
	current_period = period[current_virtual_mode];
	current_utilization = current_work / current_period;
	percentage_workload = current_work / max_work;
	current_processors_A = previous_processors_A;
	current_processors_B = previous_processors_B;
	current_processors_C = previous_processors_C;
	current_processors_D = previous_processors_D;

}

void TaskData::set_current_virtual_mode(int new_mode, bool disable)
{
	if (new_mode >= 0 && new_mode < num_modes){

		//stash old mode
		previous_mode = current_virtual_mode;

		//update CPU parameters
		current_virtual_mode = new_mode;
		current_work = work[current_virtual_mode];
		current_span = span[current_virtual_mode];
		current_period = period[current_virtual_mode];
		current_utilization = current_work / current_period;
		percentage_workload = current_work / max_work;
		previous_processors_A = current_processors_A;
		current_processors_A = processors_A[current_virtual_mode];
		
		//update GPU parameters
		previous_processors_B = current_processors_B;
		current_processors_B = processors_B[current_virtual_mode];

		//update CPU C parameters
		previous_processors_C = current_processors_C;
		current_processors_C = processors_C[current_virtual_mode];

		//update GPU D parameters
		previous_processors_D = current_processors_D;
		current_processors_D = processors_D[current_virtual_mode];

		//update the changeable flag
		changeable = (disable) ? false : true;

		//set the current mode notation to something the task actually can use
		real_current_mode = get_real_mode(current_virtual_mode);

	}

	else{
		print_module::print(std::cerr, "Error: Task ", get_index(), " was told to go to invalid mode ", new_mode, ". Ignoring.\n");
	}
}

int TaskData::get_current_virtual_mode(){
	return current_virtual_mode;
}

void TaskData::reset_changeable(){
	changeable = true;
}

int TaskData::get_permanent_processor_A(){
	return permanent_processor_A;
}
	
void TaskData::set_permanent_processor_A(int perm){
	permanent_processor_A = perm;
}

int TaskData::get_num_adaptations(){
	return num_adaptations;
}

void TaskData::set_num_adaptations(int new_num){
	num_adaptations = new_num;
}

timespec TaskData::get_work(int index){
	return work[index];
}

timespec TaskData::get_span(int index){
	return span[index];
}

timespec TaskData::get_period(int index){
	return period[index];
}

int TaskData::get_processors_A(int index){
	return processors_A[index];
}

bool TaskData::pure_A_task(){
	return is_pure_A_task;
}

int TaskData::get_original_modes_passed(){
	return modes_originally_passed;
}

void TaskData::set_processors_A_change(int num_processors_A_to_return){
	processors_A_to_return = num_processors_A_to_return;
}

void TaskData::set_processors_B_change(int num_processors_B_to_return){
	processors_B_to_return = num_processors_B_to_return;
}

void TaskData::set_processors_C_change(int num_processors_C_to_return){
	processors_C_to_return = num_processors_C_to_return;
}

void TaskData::set_processors_D_change(int num_processors_D_to_return){
	processors_D_to_return = num_processors_D_to_return;
}

int TaskData::get_processors_A_change(){
	return processors_A_to_return;
}

int TaskData::get_processors_B_change(){
	return processors_B_to_return;
}

int TaskData::get_processors_C_change(){
	return processors_C_to_return;
}

int TaskData::get_processors_D_change(){
	return processors_D_to_return;
}

bool TaskData::check_mode_transition(){
	return mode_transitioned;
}

void TaskData::set_mode_transition(bool state){
	mode_transitioned = state;
}

int TaskData::pop_back_processor_A(){

    // Handle empty vector case
    if (processor_A_mask == 0) {
        return -1;
    }

    int msb = 127;  // Start from highest possible bit
    __uint128_t test_bit = (__uint128_t)1 << 127;

    // Find the most significant 1 bit
    while ((processor_A_mask & test_bit) == 0) {
        msb--;
        test_bit >>= 1;
    }

    // Clear the MSB
    processor_A_mask ^= test_bit;

    return msb;
}

int TaskData::pop_back_processor_B(){
	
	// Handle empty vector case
    if (processor_B_mask == 0) {
        return -1;
    }

    int msb = 127;  // Start from highest possible bit
    __uint128_t test_bit = (__uint128_t)1 << 127;

    // Find the most significant 1 bit
    while ((processor_B_mask & test_bit) == 0) {
        msb--;
        test_bit >>= 1;
    }

    // Clear the MSB
    processor_B_mask ^= test_bit;

    return msb;
}

int TaskData::push_back_processor_A(int value){
    // Check if value is in valid range
    if (value < 0 || value > 127) {
        return false;
    }
    
    // Check if bit is already set
    __uint128_t bit = (__uint128_t)1 << value;
    if (processor_A_mask & bit) {
        return false;
    }
    
    // Set the bit
    processor_A_mask |= bit;

    return true;
}

int TaskData::push_back_processor_B(int value){
    // Check if value is in valid range
    if (value < 0 || value > 127) {
        return false;
    }
    
    // Check if bit is already set
    __uint128_t bit = (__uint128_t)1 << value;
    if (processor_B_mask & bit) {
        return false;
    }
    
    // Set the bit
    processor_B_mask |= bit;

    return true;
}

int TaskData::pop_back_processor_C(){

    // Find the highest set bit and clear it
    for (int i = 127; i >= 0; i--) {
        __uint128_t bit = (__uint128_t)1 << i;
        if (processor_C_mask & bit) {
            processor_C_mask &= ~bit;
            return i;
        }
    }
    return -1; // No bits set
}

int TaskData::push_back_processor_C(int value){
    // Check if value is in valid range
    if (value < 0 || value > 127) {
        return false;
    }
    
    // Check if bit is already set
    __uint128_t bit = (__uint128_t)1 << value;
    if (processor_C_mask & bit) {
        return false;
    }
    
    // Set the bit
    processor_C_mask |= bit;

    return true;
}

int TaskData::pop_back_processor_D(){

    // Find the highest set bit and clear it
    for (int i = 127; i >= 0; i--) {
        __uint128_t bit = (__uint128_t)1 << i;
        if (processor_D_mask & bit) {
            processor_D_mask &= ~bit;
            return i;
        }
    }
    return -1; // No bits set
}

int TaskData::push_back_processor_D(int value){
    // Check if value is in valid range
    if (value < 0 || value > 127) {
        return false;
    }
    
    // Check if bit is already set
    __uint128_t bit = (__uint128_t)1 << value;
    if (processor_D_mask & bit) {
        return false;
    }
    
    // Set the bit
    processor_D_mask |= bit;

    return true;
}

std::vector<int> TaskData::get_processor_A_owned_by_process(){
    std::vector<int> result;
    result.reserve(128);
    
    for (int i = 0; i < 128; i++) {
        if (processor_A_mask & ((__uint128_t)1 << i)) {

            result.push_back(i);

        }
    }
    return result;
}

std::vector<int> TaskData::get_processor_B_owned_by_process(){

	//loop over our processor B mask and if a bit is set to 1, add it to the vector
	std::vector<int> processors_B_owned_by_task;

	for (int i = 0; i < 128; i++){

		if (processor_B_mask & ((__uint128_t)1 << i)) {

			processors_B_owned_by_task.push_back(i);

		}
	}

	return processors_B_owned_by_task;

}

std::vector<int> TaskData::get_processor_C_owned_by_process(){
    std::vector<int> result;
    result.reserve(128);
    
    for (int i = 0; i < 128; i++) {
        if (processor_C_mask & ((__uint128_t)1 << i)) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<int> TaskData::get_processor_D_owned_by_process(){

	//loop over our processor D mask and if a bit is set to 1, add it to the vector
	std::vector<int> processors_D_owned_by_task;

	for (int i = 0; i < 128; i++){

		if (processor_D_mask & ((__uint128_t)1 << i)) {

			processors_D_owned_by_task.push_back(i);

		}
	}

	return processors_D_owned_by_task;

}

//retrieve the number of processor A or processor B we have been given	
std::vector<std::pair<int, std::vector<int>>> TaskData::get_processors_A_granted_from_other_tasks(){

	std::vector<std::pair<int, std::vector<int>>> returning_processors_A_granted_from_other_tasks;

	//stupid conversion to make the vectors
	for (int i = 0; i < MAXTASKS + 1; i++){
		if (processors_A_granted_from_other_tasks[i][0] != -1){

			auto current = std::make_pair(i, std::vector<int>());

			for (int j = 1; j < processors_A_granted_from_other_tasks[i][0] + 1; j++)
				current.second.push_back(processors_A_granted_from_other_tasks[i][j]);

			returning_processors_A_granted_from_other_tasks.push_back(current);

		}

	}

	return returning_processors_A_granted_from_other_tasks;

}

std::vector<std::pair<int, std::vector<int>>> TaskData::get_processors_B_granted_from_other_tasks(){

	std::vector<std::pair<int, std::vector<int>>> result;

	for (int i = 0; i < MAXTASKS + 1; i++){

		if (processors_B_granted_from_other_tasks[i][0] != -1){

			std::vector<int> processors_B;

			for (int j = 1; j < NUM_PROCESSOR_B + 1; j++){

				if (processors_B_granted_from_other_tasks[i][j] != -1){

					processors_B.push_back(processors_B_granted_from_other_tasks[i][j]);

				}

			}

			result.push_back(std::make_pair(i, processors_B));

		}

	}

	return result;

}

std::vector<std::pair<int, std::vector<int>>> TaskData::get_processors_C_granted_from_other_tasks(){

	std::vector<std::pair<int, std::vector<int>>> result;

	for (int i = 0; i < MAXTASKS + 1; i++){

		if (processors_C_granted_from_other_tasks[i][0] != -1){

			std::vector<int> processors_C;

			for (int j = 1; j < NUM_PROCESSOR_C + 1; j++){

				if (processors_C_granted_from_other_tasks[i][j] != -1){

					processors_C.push_back(processors_C_granted_from_other_tasks[i][j]);

				}

			}

			result.push_back(std::make_pair(i, processors_C));

		}

	}

	return result;

}

std::vector<std::pair<int, std::vector<int>>> TaskData::get_processors_D_granted_from_other_tasks(){

	std::vector<std::pair<int, std::vector<int>>> result;

	for (int i = 0; i < MAXTASKS + 1; i++){

		if (processors_D_granted_from_other_tasks[i][0] != -1){

			std::vector<int> processors_D;

			for (int j = 1; j < NUM_PROCESSOR_D + 1; j++){

				if (processors_D_granted_from_other_tasks[i][j] != -1){

					processors_D.push_back(processors_D_granted_from_other_tasks[i][j]);

				}

			}

			result.push_back(std::make_pair(i, processors_D));

		}

	}

	return result;

}

//give CPUs or GPUs to another task

/*****************************************************************************************


If a task is giving up CPUs, keep track of what tasks it gives to in a table.
When all tasks have been processed, send a message from the scheduler of type -1 to 
the first queue with the message id set to the task index in question. 

All other tasks will read from this queue when scheduler signals that it is time to
transition, and any process which gives up cpus will send a message back into that same
queue with the message id set to the task index in question.

2 System V Message Queues are used to facilitate this process.

Queue 1: Used to signal that a task is giving up CPUs. The message type is the task index

Queue 2: Id determines process target. Message contents are the CPUs or GPUs to be given by the processor number

Queue 3: Used to to signal that a task should give up the resources contained in the mask to the task specified

struct queue_one_message {
    long int    mtype;
	__uint128_t tasks_giving_processors;
}

struct queue_two_message {
	long int    mtype;
	long int   giving_task;
	long int   processor_type;
	__uint128_t processors;
}

struct queue_three_message {
	long int    mtype;
	long int   task_index;
	long int   processor_type;
	long int   processor_ct;
}

Structure:

If Time to Reschedule:

	Read From queue 1 for message of id == task_index

		- If message type == -1 : global_variable tasks_to_listen_for = message.tasks_giving_processors

	If all tasks_to_listen_for == 0, then we can transition

		For each read from Queue 2

			If processor_type == CPU : CPU_mask |= processors
			If processor_type == GPU : TPC_mask |= processors

		OR these values with the current masks

		Read from Queue 3

			If processor_type == CPU : CPU_mask &= ~processors
			If processor_type == GPU : TPC_mask &= ~processors

			send message to queue 2 with the processors to be given



		Transition complete


******************************************************************************************/

void TaskData::set_processors_to_send_to_other_processes(int task_to_send_to, int processor_type, int processor_ct){

	//send message into queue 3
	struct queue_three_message message;

	message.mtype = get_index() + 1;
	message.task_index = task_to_send_to + 1;
	message.processor_type = processor_type;
	message.processor_ct = processor_ct;

	//std::cerr<< "Telling task " << get_index() << " to give " << processor_ct << " processors to task " << task_to_send_to << ".\n";

	if (msgsnd(queue_three, &message, sizeof(message) - sizeof(long), 0) == -1){

		print_module::print(std::cerr, "Error: Failed to send a message to queue 3: failed with: ", strerror(errno), " \n");
		kill(0, SIGTERM);

	}
	
}

void TaskData::set_tasks_to_wait_on(__uint128_t task_mask){	

	//send message into queue 1
	struct queue_one_message message;

	message.mtype = get_index() + 1;
	message.tasks_giving_processors = task_mask;

	if (msgsnd(queue_one, &message, sizeof(message) - sizeof(long), 0) == -1){

		print_module::print(std::cerr, "Error: Failed to send message to queue 1.\n");
		kill(0, SIGTERM);

	}
	
}

void TaskData::start_transition(){

	//read from queue 1 
	struct queue_one_message message;

	if (msgrcv(queue_one, &message, sizeof(message) - sizeof(long), get_index() + 1, IPC_NOWAIT) != -1){
		
		tasks_giving_processors = message.tasks_giving_processors;

	}

}

//fetch any processors that have been given to us
//and returns true when all the resources have been given
//to us from other processes: NON BLOCKING
bool TaskData::get_processors_granted_from_other_tasks(){

	//read from queue 2, until we have no messages available
	struct queue_two_message message;

	while (msgrcv(queue_two, &message, sizeof(message) - sizeof(long), get_index() + 1, IPC_NOWAIT) != -1){

		//if the message is for CPUs
		if (message.processor_type == 0)
			processors_A_received |= message.processors;

		//if the message is for GPUs
		else if (message.processor_type == 1)
			processors_B_received |= message.processors;

		//if the message is for CPUs C
		else if (message.processor_type == 2)
			processors_C_received |= message.processors;

		//if the message is for GPUs D
		else if (message.processor_type == 3)
			processors_D_received |= message.processors;

		tasks_giving_processors &= ~((__uint128_t)(1) << message.giving_task);

	}

	//std::cerr<< "Task " << get_index() << " has received " << (unsigned long long) processors_A_received << " CPUs and " << (unsigned long long) processors_B_received << " GPUs.\n";

	//if (tasks_giving_processors != 0)
		//std::cerr<< "Task " << get_index() << " is still waiting on " << (unsigned long long) tasks_giving_processors << " tasks.\n";

	return tasks_giving_processors == 0;

}

void TaskData::set_processors_A_to_send_to_other_processes(std::pair<int, int> entry){

	//send message into queue 2
	struct queue_two_message message;

	message.mtype = entry.first;
	message.giving_task = get_index();
	message.processor_type = 0;
	message.processors = 0;

	//always grabs from the back of the vector
	for (int i = 0; i < entry.second; i++){

		int processors_A_selected_to_send = pop_back_processor_A();

		message.processors |= ((__uint128_t)1 << processors_A_selected_to_send);
	}

	if (msgsnd(queue_two, &message, sizeof(message) - sizeof(long), 0) == -1){

		print_module::print(std::cerr, "Error: Failed to send message to queue 2.\n");
		kill(0, SIGTERM);

	}

}

void TaskData::set_processors_B_to_send_to_other_processes(std::pair<int, int> entry){

	//send message into queue 2
	struct queue_two_message message;

	message.mtype = entry.first;
	message.giving_task = get_index();
	message.processor_type = 1;
	message.processors = 0;

	//always grabs from the back of the vector
	for (int i = 0; i < entry.second; i++){

		int processors_B_selected_to_send = pop_back_processor_B();

		//std::cerr << "Task " << get_index() << " is sending processor B " << processors_B_selected_to_send << " to task " << entry.first << ".\n";

		message.processors |= ((__uint128_t)1 << processors_B_selected_to_send);

	}

	if (msgsnd(queue_two, &message, sizeof(message) - sizeof(long), 0) == -1){

		print_module::print(std::cerr, "Error: Failed to send message to queue 2.\n");
		kill(0, SIGTERM);

	}

}

void TaskData::set_processors_C_to_send_to_other_processes(std::pair<int, int> entry){

	//send message into queue 2
	struct queue_two_message message;

	message.mtype = entry.first;
	message.giving_task = get_index();
	message.processor_type = 2;  // C processor type
	message.processors = 0;

	//always grabs from the back of the vector
	for (int i = 0; i < entry.second; i++){

		int processors_C_selected_to_send = pop_back_processor_C();

		message.processors |= ((__uint128_t)1 << processors_C_selected_to_send);

	}

	if (msgsnd(queue_two, &message, sizeof(message) - sizeof(long), 0) == -1){

		print_module::print(std::cerr, "Error: Failed to send message to queue 2.\n");
		kill(0, SIGTERM);

	}

}

void TaskData::set_processors_D_to_send_to_other_processes(std::pair<int, int> entry){

	//send message into queue 2
	struct queue_two_message message;

	message.mtype = entry.first;
	message.giving_task = get_index();
	message.processor_type = 3;  // D processor type
	message.processors = 0;

	//always grabs from the back of the vector
	for (int i = 0; i < entry.second; i++){

		int processors_D_selected_to_send = pop_back_processor_D();

		message.processors |= ((__uint128_t)1 << processors_D_selected_to_send);

	}

	if (msgsnd(queue_two, &message, sizeof(message) - sizeof(long), 0) == -1){

		print_module::print(std::cerr, "Error: Failed to send message to queue 2.\n");
		kill(0, SIGTERM);

	}

}	

void TaskData::acquire_all_processors(){
	
	//if we have been given processors, then we can acquire them
	processor_A_mask |= processors_A_received;
	processor_B_mask |= processors_B_received;
	processor_C_mask |= processors_C_received;
	processor_D_mask |= processors_D_received;

	//clear the variables
	processors_A_received = 0;
	processors_B_received = 0;
	processors_C_received = 0;
	processors_D_received = 0;
	tasks_giving_processors = 0;

}

void TaskData::give_processors_to_other_tasks(){

	//read from queue 3 until we cant
	struct queue_three_message message;

	while (msgrcv(queue_three, &message, sizeof(message) - sizeof(long), get_index() + 1, IPC_NOWAIT) != -1){

		//if the message is for processors A
		if (message.processor_type == 0){

			set_processors_A_to_send_to_other_processes(std::make_pair(message.task_index, message.processor_ct));

			//std::cerr<< "Task " << get_index() << " is giving " << message.processor_ct << " processors A to task " << message.task_index << ".\n";

		}

		//if the message is for processors B
		else if (message.processor_type == 1){

			set_processors_B_to_send_to_other_processes(std::make_pair(message.task_index, message.processor_ct));

			//std::cerr<< "Task " << get_index() << " is giving " << message.processor_ct << " processors B to task " << message.task_index << ".\n";

		}

		//if the message is for processors C
		else if (message.processor_type == 2){

			set_processors_C_to_send_to_other_processes(std::make_pair(message.task_index, message.processor_ct));

		}

		//if the message is for processors D
		else if (message.processor_type == 3){

			set_processors_D_to_send_to_other_processes(std::make_pair(message.task_index, message.processor_ct));

		}

	}

	//std::cerr<< "Task " << get_index() << " finished giving processors.\n";

}

//make a function which clears these vectors like they are cleared in the constructor
void TaskData::clear_processors_A_granted_from_other_tasks(){

	for (size_t i = 0; i < MAXTASKS + 1; i++)
		processors_A_granted_from_other_tasks[i][0] = -1;

}

void TaskData::clear_processors_B_granted_from_other_tasks(){

	for (size_t i = 0; i < MAXTASKS + 1; i++)
		processors_B_granted_from_other_tasks[i][0] = -1;

}

void TaskData::clear_processors_C_granted_from_other_tasks(){

	for (size_t i = 0; i < MAXTASKS + 1; i++)
		processors_C_granted_from_other_tasks[i][0] = -1;

}

void TaskData::clear_processors_D_granted_from_other_tasks(){

	for (size_t i = 0; i < MAXTASKS + 1; i++)
		processors_D_granted_from_other_tasks[i][0] = -1;

}

int TaskData::get_real_current_mode(){

	//return the real current mode
	return real_current_mode;

}

void TaskData::set_real_current_mode(int new_mode, bool disable){

	if (new_mode >= 0 && new_mode < modes_originally_passed){

		//update the changeable flag
		changeable = (disable) ? false : true;

		//set the current mode notation to something the task actually can use
		real_current_mode = new_mode;

	}

	else{
		print_module::print(std::cerr, "Error: Task ", get_index(), " was told to go to invalid mode ", new_mode, ". Ignoring.\n");
	}

}

__uint128_t TaskData::get_processor_A_mask() {
	return processor_A_mask;
}

__uint128_t TaskData::get_processor_B_mask() {
	return processor_B_mask;
}

__uint128_t TaskData::get_processor_C_mask() {
	
	return processor_C_mask;

}

__uint128_t TaskData::get_processor_D_mask() {
	
	return processor_D_mask;

}

bool TaskData::is_combinatorially_elastic(){
	return combinatorially_elastic;
}

void TaskData::set_cooperative(bool state){

	cooperative_bool = state;

}

bool TaskData::cooperative(){
	
	return cooperative_bool;

}

int TaskData::get_permanent_processor_index(){

	return permanent_processor_index;

}

int TaskData::get_permanent_processor_core(){

	return permanent_processor_core;

}

int TaskData::get_previous_permanent_processor_index(){

	return previous_permanent_processor_index;

}

void TaskData::acknowledge_permanent_processor_switch(){

	previous_permanent_processor_index = permanent_processor_index;

}

void TaskData::set_permanent_processor_index(int index){

	permanent_processor_index = index;

}

void TaskData::set_permanent_processor_core(int core){

	permanent_processor_core = core;
}

void TaskData::elect_permanent_processor(){

	if (permanent_processor_index != -1){
		return;
	}

	//elect the permanent processor with a preference for the lowest A core 
	for (int i = 0; i < 4; i++){
		std::vector<int> processor_vectors;
		
		switch (i){

			case 0:
				processor_vectors = get_processor_D_owned_by_process();
				break;

			case 1:
				processor_vectors = get_processor_C_owned_by_process();
				break;

			case 2:
				processor_vectors = get_processor_B_owned_by_process();
				break;

			case 3:
				processor_vectors = get_processor_A_owned_by_process();	
				break;

		}

		if (processor_vectors.size() != 0){
			set_permanent_processor_index(3 - i);
			set_permanent_processor_core(processor_vectors.at(0));
		}

	}

}