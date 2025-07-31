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

	std::vector<int> processors_equivalent_to_A;

	for (int i = 0; i < 4; i++){

		if (std::get<0>(equivalent_vector[i]) == 0)
			processors_equivalent_to_A.push_back(i);

	}
	
	//now up to the number of processors we have in the system, we can generate all possible combinations of processor
	//configurations for the "A" portion of the DAG. Use a recursive lambda function to generate all possible combinations.
	// Define the recursive function using std::function for proper self-reference
	std::function<std::vector<std::tuple<int,int,int,int, double>>(int, std::vector<int>)> generate_processor_combinations = 
	
	[&](int current_processor, std::vector<int> current_combination) -> std::vector<std::tuple<int,int,int,int, double>> {

		std::vector<std::tuple<int,int,int,int, double>> new_combinations;

		double processor_average_starting = 0;

		for (int i = 0; i < current_combination.size(); i++)
			processor_average_starting += std::get<1>(equivalent_vector[i]) * current_combination[i];

		int total_processors_used = (current_combination[0] + current_combination[1] + current_combination[2] + current_combination[3]);

		for (int i = (current_processor == 0 ? 1 : 0); i <= system_processors[current_processor]; i++){

			double new_processor_average = processor_average_starting + (std::get<1>(equivalent_vector[current_processor]) * i);

			new_processor_average /= (total_processors_used + i);

			//now check if we can complete the work in the window of time given
			double time_to_complete = L_A + ((CpA - L_A) / (new_processor_average * (total_processors_used + i)));

			if (time_to_complete > T)
				continue;

			//If we are not the last processor, then we can recurse
			if (current_processor != processors_equivalent_to_A.back()){

				std::vector<int> new_combination = current_combination;

				new_combination[current_processor] = i;

				auto new_combinations_from_recurse = generate_processor_combinations(current_processor + 1, new_combination);
				new_combinations.insert(new_combinations.end(), new_combinations_from_recurse.begin(), new_combinations_from_recurse.end());

			}

			else {

				//if we make it down here, then we can add this to the combinations list
				new_combinations.push_back(std::make_tuple(current_combination[0] + (current_processor == 0 ? i : 0), current_combination[1] + (current_processor == 1 ? i : 0), current_combination[2] + (current_processor == 2 ? i : 0), current_combination[3] + (current_processor == 3 ? i : 0), time_to_complete));

			}

		}

		return new_combinations;

	};

	//generate all possible combinations of processors for the "A" portion of the DAG
	auto processor_combinations = generate_processor_combinations(0, std::vector<int>(4, 0));

    std::vector<std::tuple<int,int,int,int>> result;

	//determine which processor is not in the equivalent vector
	int processor_not_in_equivalent_vector = -1;
	for (int i = 0; i < 4; i++){
		if (std::get<0>(equivalent_vector[i]) != 0){
			processor_not_in_equivalent_vector = i;
			break;
		}
	}

	if (processor_not_in_equivalent_vector == -1){

		//move all the combinations to the result vector
		for (auto combination : processor_combinations)
			result.push_back(std::make_tuple(std::get<0>(combination), std::get<1>(combination), std::get<2>(combination), std::get<3>(combination)));

		return result;
	
	}

	else {
    
		//For all the modes we just made, we can now generate all possible combinations of processors for the "B" portion of the DAG
		for (auto combination : processor_combinations){

			//For each combination, determine how much of the period is left for the "B" portion of the DAG
			double period_left = T - std::get<4>(combination);

			//If the period left is less than the span of the "B" portion of the DAG, then we can skip this combination
			if (period_left <= processor_information[processor_not_in_equivalent_vector].second)
				continue;

			//Otherwise, determine what the minimum number of processors needed for the "B" portion of the DAG is
			int minimum_processors_needed_for_B = std::ceil((processor_information[processor_not_in_equivalent_vector].first - processor_information[processor_not_in_equivalent_vector].second) / (period_left - processor_information[processor_not_in_equivalent_vector].second));

			//If the minimum number of processors needed for the "B" portion of the DAG is greater than the number of processors available, then we can skip this combination
			if (minimum_processors_needed_for_B > system_processors[processor_not_in_equivalent_vector])
				continue;

			//otherwise make an entry in the result vector
			result.push_back(std::make_tuple(std::get<0>(combination) + (0 == processor_not_in_equivalent_vector ? minimum_processors_needed_for_B : 0), std::get<1>(combination) + (1 == processor_not_in_equivalent_vector ? minimum_processors_needed_for_B : 0), std::get<2>(combination) + (2 == processor_not_in_equivalent_vector ? minimum_processors_needed_for_B : 0), std::get<3>(combination) + (3 == processor_not_in_equivalent_vector ? minimum_processors_needed_for_B : 0)));
			
		}

	}

	if (result.empty()){
		print_module::print(std::cerr, "Error: No valid allocations found for task mode with parameters ", CpA, " ", L_A, " ", CpB, " ", L_B, " ", T, "\n");
		exit(-1);
	}

	//now we need to filter out the modes which are not the minimum number of processors needed for the A and B portions of the DAG
	std::vector<std::tuple<int,int,int,int>> result_filtered;

	for (int i = 0; i < result.size(); i++){

		bool add = true;

		//using the equivalent vector, if the mode has 0 for all the processors, then we can skip it
		bool all_zero = true;
		if (std::get<0>(equivalent_vector[0]) == 0)
			if (std::get<0>(result[i]) != 0)
				all_zero = false;
		if (std::get<0>(equivalent_vector[1]) == 0)
			if (std::get<1>(result[i]) != 0)
				all_zero = false;
		if (std::get<0>(equivalent_vector[2]) == 0)
			if (std::get<2>(result[i]) != 0)
				all_zero = false;
		if (std::get<0>(equivalent_vector[3]) == 0)
			if (std::get<3>(result[i]) != 0)
				all_zero = false;

		if (all_zero)
			continue;

		for (int j = 0; j < i; j++){

			if (i != j){

				if ((std::get<0>(result[i]) > std::get<0>(result[j])) && (std::get<1>(result[i]) == std::get<1>(result[j])) && (std::get<2>(result[i]) == std::get<2>(result[j])) && (std::get<3>(result[i]) == std::get<3>(result[j])) ||
					(std::get<0>(result[i]) == std::get<0>(result[j])) && (std::get<1>(result[i]) > std::get<1>(result[j])) && (std::get<2>(result[i]) == std::get<2>(result[j])) && (std::get<3>(result[i]) == std::get<3>(result[j])) ||
					(std::get<0>(result[i]) == std::get<0>(result[j])) && (std::get<1>(result[i]) == std::get<1>(result[j])) && (std::get<2>(result[i]) > std::get<2>(result[j])) && (std::get<3>(result[i]) == std::get<3>(result[j])) ||
					(std::get<0>(result[i]) == std::get<0>(result[j])) && (std::get<1>(result[i]) == std::get<1>(result[j])) && (std::get<2>(result[i]) == std::get<2>(result[j])) && (std::get<3>(result[i]) > std::get<3>(result[j]))){

					add = false;
					break;

				}

			}

		}

		if (add)
			result_filtered.push_back(result[i]);

	}

    return result_filtered;
    
}

TaskData::TaskData(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, 
														timespec * processor_B_work_, timespec * processor_B_span_, timespec * processor_B_period_, 
														timespec * processor_C_work_, timespec * processor_C_span_, timespec * processor_C_period_, 
														timespec * processor_D_work_, timespec * processor_D_span_, timespec * processor_D_period_, bool safe) : 	
																													
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

		//for testing just set the equivalent vector to this:
		std::tuple<int, float> equivalent_vector[4] = { {0, 1}, {0, 0.75}, {2, 1}, {3, 1} };

		//pass to the computeModeResources function
		auto resources = computeModeResources(work_long, span_long, processor_B_work_long, processor_B_span_long, period_long, processor_C_work_long, processor_C_span_long, processor_D_work_long, processor_D_span_long, equivalent_vector);
		mode_options += resources.size();

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

	//check if it's our permanent
	if (msb == get_permanent_processor_A()){
		//skip the current bit
		msb--;
		test_bit >>= 1;

		while ((processor_A_mask & test_bit) == 0) {
			msb--;
			test_bit >>= 1;
		}
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
			//do not allow our permanent processor A to be returned as
			//a processor A we can pass or keep
			if (i != get_permanent_processor_A())
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