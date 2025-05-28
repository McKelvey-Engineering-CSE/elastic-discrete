#ifdef __NVCC__

	void Scheduler::create_scheduler_stream(){

		CUdevResource initial_resources;
		unsigned int partition_num = 2;
		CUdevResource resources[partition_num];

		//device specs
		CUdevResourceDesc device_resource_descriptor;

		//fill the initial descriptor
		CUDA_SAFE_CALL(cuDeviceGetDevResource(0, &initial_resources, CU_DEV_RESOURCE_TYPE_SM));

		//take the previous element above us and split it 
		//fill the corresponding portions of the matrix as we go
		CUDA_SAFE_CALL(cuDevSmResourceSplitByCount(resources, &partition_num, &initial_resources, NULL, CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));

		//now set aside the first position and make a green context from it
		CUDA_SAFE_CALL(cuDevResourceGenerateDesc(&device_resource_descriptor, &resources[0], 1));
		CUDA_SAFE_CALL(cuGreenCtxCreate(&green_ctx, device_resource_descriptor, 0, CU_GREEN_CTX_DEFAULT_STREAM));

		CUDA_SAFE_CALL(cuGreenCtxStreamCreate(&scheduler_stream, green_ctx, CU_STREAM_NON_BLOCKING, 0));
		CUDA_SAFE_CALL(cuGreenCtxStreamCreate(&cautious_stream, green_ctx, CU_STREAM_NON_BLOCKING, 0));

	}

#endif

std::vector<std::vector<Scheduler::task_mode>> Scheduler::task_table(100, std::vector<task_mode>(100));

class Schedule * Scheduler::get_schedule(){
	return &schedule;
}

int Scheduler::get_num_tasks(){
	return task_table.size();
}

void Scheduler::generate_unsafe_combinations(size_t maxCPU){}

TaskData * Scheduler::add_task(double elasticity_,  int num_modes_, timespec * work_, timespec * span_, timespec * period_, timespec * gpu_work_, timespec * gpu_span_, timespec * gpu_period_, bool safe){

	//add the task to the legacy schedule object, but also add to vector
	//to make the scheduler much easier to read and work with.
	auto taskData_object = schedule.add_task(elasticity_, num_modes_, work_, span_, period_, gpu_work_, gpu_span_, gpu_period_, safe);

	task_table.push_back(std::vector<task_mode>());

	std::cout << "Task Losses:" << std::endl; 

	int total_modes = taskData_object->get_number_of_modes();

	for (int j = 0; j < total_modes; j++){

		task_mode item;

		//the loss function is different if the 
		//task is a pure cpu task or hybrid task
		if (taskData_object->pure_cpu_task())
			item.cpuLoss = (1.0 / taskData_object->get_elasticity() * (std::pow(taskData_object->get_max_utilization() - (taskData_object->get_work(j) / taskData_object->get_period(j)), 2)));// * 1000;
		
		else 
			item.cpuLoss = (1.0 / taskData_object->get_elasticity() * (std::pow(taskData_object->get_max_utilization() - ((taskData_object->get_work(j) / taskData_object->get_period(j)) + (taskData_object->get_GPU_work(j) / taskData_object->get_period(j))), 2)));// * 1000;

		std::cout << "Mode "<< j << " Loss: " << item.cpuLoss << " Cores: " << taskData_object->get_CPUs(j) << " SMS: " << taskData_object->get_GPUs(j) << std::endl;

		item.gpuLoss = 0;
		item.cores = taskData_object->get_CPUs(j);
		item.sms = taskData_object->get_GPUs(j);

		task_table.at(task_table.size() - 1).push_back(item);

		//check all other modes stored for this task
		//and if it gains one resource while losing another
		//mark it as unsafe
		for (int i = 0; i < (int)task_table.at(task_table.size() - 1).size(); i++){

			for (int k = i; k < (int)task_table.at(task_table.size() - 1).size(); k++){

				if (task_table.at(task_table.size() - 1).at(i).cores < task_table.at(task_table.size() - 1).at(k).cores && task_table.at(task_table.size() - 1).at(i).sms > task_table.at(task_table.size() - 1).at(k).sms)
					task_table.at(task_table.size() - 1).at(i).unsafe_mode = true;

				else if (task_table.at(task_table.size() - 1).at(i).cores > task_table.at(task_table.size() - 1).at(k).cores && task_table.at(task_table.size() - 1).at(i).sms < task_table.at(task_table.size() - 1).at(k).sms)
					task_table.at(task_table.size() - 1).at(i).unsafe_mode = true;

			}

		}
		
	}

	return taskData_object;
}

/*************************************************************

Implements the scheduling algorithm; either runs the knapsack
algorithm or the cautious scheduler itself or if CUDA is
enabled, runs the CUDA version of the scheduler. It also builds
the RAG and executes it if a solution is found

*************************************************************/
void Scheduler::do_schedule(size_t maxCPU, bool check_max_possible){

	//If we compiled with CUDA enabled, we need 
	//to initialize the CUDA context and stream
	#ifdef __NVCC__

		if (first_time) {

			CUDA_SAFE_CALL(cuInit(0));

			create_scheduler_stream();

			CUDA_SAFE_CALL(cuCtxFromGreenCtx(&primary_scheduler_context, green_ctx));
			
			//set current
			CUDA_SAFE_CALL(cuCtxSetCurrent(primary_scheduler_context));

		}

	#endif

	if (first_time) {

		for (size_t i = 0; i < task_table.size(); i++){

			double worst_mode = 0;

			for (size_t j = 0; j < task_table.at(i).size(); j++){

				if (task_table.at(i).at(j).cpuLoss > worst_mode)
					worst_mode = task_table.at(i).at(j).cpuLoss;

			}

			max_loss += worst_mode;
		}

		print_module::print(std::cerr, "Max Possible Loss: ", max_loss, "\n");

	}

	
	std::vector<int> transitioned_tasks;

	//If this is the first time running the scheduler, we need to
	//initialize the DP table and the task table both on the host
	//as well as the device
	if (first_time) {

		//add an entry for each task into previous modes
		for (int i = 0; i < schedule.count(); i++)
			previous_modes.push_back(task_mode());

		//MAXTASKS tasks, 4 modes (0-3): cores, sms, cpuLoss
		int host_task_table[MAXTASKS * MAXMODES * 3];
		float host_losses[MAXTASKS * MAXMODES];

		//find largest number of modes in the task table
		int max_modes = 0;

		for (int i = 0; i < (int) task_table.size(); i++)
			if ((int) task_table.at(i).size() > max_modes)
				max_modes = (int) task_table.at(i).size();

		for (int i = 0; i < (int) task_table.size(); i++){

			for (int j = 0; j < (int) task_table.at(i).size(); j++){

				host_task_table[i * MAXMODES * 3 + j * 3 + 0] = task_table.at(i).at(j).cores;
				host_task_table[i * MAXMODES * 3 + j * 3 + 1] = task_table.at(i).at(j).sms;

				//set the real mode to the mode number
				host_task_table[i * MAXMODES * 3 + j * 3 + 2] = (schedule.get_task(i))->get_real_mode(j);

				host_losses[i * MAXMODES + j] = task_table.at(i).at(j).cpuLoss;

			}

			//if this task had fewer modes than max, pad all the rest with -1
			for (int j = (int) task_table.at(i).size(); j < MAXMODES; j++){

				host_task_table[i * MAXMODES * 3 + j * 3 + 0] = -1;
				host_task_table[i * MAXMODES * 3 + j * 3 + 1] = -1;
				host_task_table[i * MAXMODES * 3 + j * 3 + 2] = -1;
				host_losses[i * MAXMODES + j] = -1;

			}

		}

		//get the symbol on the device
		#ifdef __NVCC__

			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&d_uncooperative_tasks, sizeof(int) * MAXTASKS));
			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&d_final_solution, sizeof(int) * MAXTASKS));
			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&cautious_d_final_solution, sizeof(int) * MAXTASKS));
			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&d_losses, sizeof(double) * MAXTASKS * MAXMODES));
			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&d_final_loss, sizeof(double)));
			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&cautious_d_final_loss, sizeof(double)));
			CUDA_NEW_SAFE_CALL(cudaMalloc((void **)&d_current_task_modes, sizeof(int) * MAXTASKS * 2));

		#else 

			d_uncooperative_tasks = (int*)malloc(sizeof(int) * MAXTASKS);
			d_final_solution = (int*)malloc(sizeof(int) * MAXTASKS);
			d_current_task_modes = (int*)malloc(sizeof(int) * MAXTASKS * 2);
			d_losses = (double*)malloc(sizeof(double) * MAXTASKS * MAXMODES);
			d_final_loss = (double*)malloc(sizeof(double));

		#endif

		//copy it
		#ifdef __NVCC__

			CUDA_NEW_SAFE_CALL(cudaMemcpy(d_losses, host_losses, sizeof(float) * MAXTASKS * MAXMODES, cudaMemcpyHostToDevice));

			CUDA_NEW_SAFE_CALL(cudaMemcpyToSymbol(constant_task_table, &host_task_table, sizeof(int) * MAXTASKS * MAXMODES * 3));
			CUDA_NEW_SAFE_CALL(cudaMemcpyToSymbol(constant_losses, &host_losses, sizeof(float) * MAXTASKS * MAXMODES));

		#else 

			memcpy(d_losses, host_losses, sizeof(float) * MAXTASKS * MAXMODES);

			memcpy(constant_task_table, host_task_table, sizeof(int) * MAXTASKS * MAXMODES * 3);
			memcpy(constant_losses, host_losses, sizeof(float) * MAXTASKS * MAXMODES);

		#endif

	}

	int N = task_table.size();
	std::vector<int> best_solution;

	//This segment is a forced core and sm (core_A or core_B)
	//check to ensure that all tasks actually have the cores 
	//that they are supposed to within their TaskData masks.
	//If they do not, we do not try to recover from whatever
	//went wrong, we just error out.
	if (!first_time){

		int total_cores = 0;
		int total_gpus = 0;

		//collect all the resources that the free pool was given
		//via it's TaskData entry in the scheduler
		schedule.get_task(previous_modes.size())->get_processors_granted_from_other_tasks();
		schedule.get_task(previous_modes.size())->acquire_all_processors();

		//for each task, check that the number of cores and gpus
		//that the task has is the same as the number of cores and
		//gpus that the task is supposed to have
		//(-1 because we always give one to the scheduler)
		for (int i = 0; i < (int) previous_modes.size(); i++){
			
			auto task_owned_cpus = (schedule.get_task(i))->get_cpu_owned_by_process();

			if (((previous_modes.at(i).cores - 1) != (int) task_owned_cpus.size())){
				
				std::cout << "CPU Count Mismatch. Process:" << i << " | Cores assigned: " << previous_modes.at(i).cores << " | Cores found: " << task_owned_cpus.size() << " | Cannot Continue" << std::endl;
				killpg(process_group, SIGINT);
				return;

			}
			
			auto task_owned_gpus = (schedule.get_task(i))->get_gpu_owned_by_process();

			if ((previous_modes.at(i).sms) != (int) task_owned_gpus.size()){

				std::cout << "GPU Count Mismatch. Process:" << i << " | GPUs assigned: " << previous_modes.at(i).sms << " | GPUs found: " << task_owned_gpus.size() << " | Cannot Continue" << std::endl;
				killpg(process_group, SIGINT);
				return;

			}

			//add to total
			total_cores += previous_modes.at(i).cores;
			total_gpus += previous_modes.at(i).sms;

		}

		//check that the total cores in the system - the total found is the free count
		if (((int) maxCPU - total_cores) != (int) schedule.get_task(previous_modes.size())->get_cpu_owned_by_process().size()){

			std::cout << "CPU Count Mismatch. Total Cores: " << maxCPU << " | Total Found: " << total_cores << " | Free Cores: " << std::bitset<128>(schedule.get_task(previous_modes.size())->get_cpu_mask()).count() << " | Cannot Continue" << std::endl;
			killpg(process_group, SIGINT);
			return;

		}

		//check that the total gpus in the system - the total found is the free count
		if (((int) NUMGPUS - total_gpus) != (int) schedule.get_task(previous_modes.size())->get_gpu_owned_by_process().size()){

			std::cout << "GPU Count Mismatch. Total GPUs: " << NUMGPUS << " | Total Found: " << total_gpus << " | Free GPUs: " << std::bitset<128>(schedule.get_task(previous_modes.size())->get_gpu_mask()).count() << " | Cannot Continue" << std::endl;
			killpg(process_group, SIGINT);
			return;

		}

	}

	//get current time
	timespec start_time, end_time;
	clock_gettime(CLOCK_MONOTONIC, &start_time);

	double loss;
	double elapsed_time;

	//loop over all the tasks and determine which
	//ones are set to be uncooperative. An uncooperative
	//task is either transitioning or has just permanently
	//set itself to be uncooperative. Either way, these tasks
	//are not allowed to change their modes in the knapsack
	//algorithm. We take note of which tasks are uncooperative
	//and send them to the device
	int host_uncooperative[MAXTASKS] = {-1};
	memset(host_uncooperative, -1, sizeof(int) * MAXTASKS);

	//loopback variables to ensure we process
	//the uncooperative tasks last every time
	int real_order_of_tasks[num_tasks];
	int loopback_back = num_tasks - 1;
	int loopback_front = 0;

	for (int i = 0; i < schedule.count(); i++){

		if (!(schedule.get_task(i))->get_changeable() || !(schedule.get_task(i))->cooperative()){
				
			if (!first_time)
				host_uncooperative[i] = schedule.get_task(i)->get_real_current_mode();

			real_order_of_tasks[loopback_back--] = i;

		}

		else 
			real_order_of_tasks[loopback_front++] = i;

	}

	//copy over the current state of the task system
	int host_current_modes[MAXTASKS * 2];
	memset(host_current_modes, 0, sizeof(int) * MAXTASKS * 2);

	if (!first_time){

		for (size_t i = 0; i < previous_modes.size(); i++){

			host_current_modes[i * 2] = previous_modes.at(i).cores;
			host_current_modes[i * 2 + 1] = previous_modes.at(i).sms;

		}

	}

	//If CUDA is enabled, copy all the data that we need for
	//each run of the knapsack algorithm to the device
	//and actually run the knapsack algorithm on the device
	//if CUDA is not enabled, just run the knapsack algorithm
	//on the host

	int slack_A = maxCPU;
	int slack_B = NUMGPUS;

	if (first_time) {
		
		#ifdef __NVCC__

		CUDA_NEW_SAFE_CALL(cudaFuncSetAttribute(device_do_schedule,
						cudaFuncAttributeMaxDynamicSharedMemorySize,
						66 * 65 * 3 * 4));

		#endif
	
	}

	else {

		for (int i = 0; i < (int) previous_modes.size() * 2; i += 2){
			
			slack_A -= host_current_modes[i];
			slack_B -= host_current_modes[i + 1];

		}

	}

	pm::print(std::cerr, "[Starting Slack] Slack A: ", slack_A, " Slack B: ", slack_B, "\n");

	#ifdef __NVCC__

		CUDA_NEW_SAFE_CALL(cudaMemcpy(d_current_task_modes, host_current_modes, sizeof(int) * MAXTASKS * 2, cudaMemcpyHostToDevice));
		CUDA_NEW_SAFE_CALL(cudaMemcpy(d_uncooperative_tasks, host_uncooperative, MAXTASKS * sizeof(int), cudaMemcpyHostToDevice));

		//Execute exact solution
		device_do_schedule<<<1, 1024, 66 * 65 * 3 * 4, scheduler_stream>>>(N - 1, maxCPU, NUMGPUS, d_current_task_modes, d_losses, d_final_loss, d_uncooperative_tasks, d_final_solution, slack_A, slack_B, 0);

		//peek for launch errors
		CUDA_NEW_SAFE_CALL(cudaPeekAtLastError());

		//copy the final_solution array back
		int host_final[MAXTASKS] = {0};

		//copy it 
		CUDA_NEW_SAFE_CALL(cudaMemcpyAsync(host_final, d_final_solution, MAXTASKS * sizeof(int), cudaMemcpyDeviceToHost, scheduler_stream));
		CUDA_NEW_SAFE_CALL(cudaMemcpyAsync(&loss, d_final_loss, sizeof(double), cudaMemcpyDeviceToHost, scheduler_stream));

		CUDA_NEW_SAFE_CALL(cudaStreamSynchronize(scheduler_stream));

		//if we are running the scheduler twice to compare 
		//against maximum possible value for a given transition
		if (check_max_possible){

			//first check just the constrained version of the problem
			device_do_schedule<<<1, 1024, 66 * 65 * 3 * 4, scheduler_stream>>>(N - 1, maxCPU, NUMGPUS, d_current_task_modes, d_losses, d_final_loss, d_uncooperative_tasks, d_final_solution, slack_A, slack_B, 1);

			CUDA_NEW_SAFE_CALL(cudaPeekAtLastError());

			//copy the error
			double constrained_value = 0;
			CUDA_NEW_SAFE_CALL(cudaMemcpyAsync(&constrained_value, d_final_loss, sizeof(double), cudaMemcpyDeviceToHost, scheduler_stream));
			CUDA_NEW_SAFE_CALL(cudaStreamSynchronize(scheduler_stream));


			//enable unsafe checking
			int optimal_modes[MAXTASKS * 2];
			memset(optimal_modes, 0, sizeof(int) * MAXTASKS * 2);

			CUDA_NEW_SAFE_CALL(cudaMemcpy(d_current_task_modes, optimal_modes, sizeof(int) * MAXTASKS * 2, cudaMemcpyHostToDevice));

			device_do_schedule<<<1, 1024, 66 * 65 * 3 * 4, scheduler_stream>>>(N - 1, maxCPU, NUMGPUS, d_current_task_modes, d_losses, d_final_loss, d_uncooperative_tasks, d_final_solution, slack_A, slack_B, 0);

			CUDA_NEW_SAFE_CALL(cudaPeekAtLastError());

			//copy the error
			double max_possible_value = 0;
			CUDA_NEW_SAFE_CALL(cudaMemcpyAsync(&max_possible_value, d_final_loss, sizeof(double), cudaMemcpyDeviceToHost, scheduler_stream));
			CUDA_NEW_SAFE_CALL(cudaStreamSynchronize(scheduler_stream));

			double our_loss = 0;

			if (max_possible_value > 100){

				max_possible_value = 0;
				constrained_value = 0;
				our_loss = 0;

			}

			else {

				max_possible_value = (max_loss - max_possible_value) / max_loss;
				constrained_value = (max_loss - constrained_value) / max_loss;
				our_loss = (max_loss - loss) / max_loss;

			}



			//print the percentage our values are worse than optimal
			pm::print(std::cerr, "Amount the constrained result worse than the optimal system state: ", max_possible_value - constrained_value, "\n");
			pm::print(std::cerr, "Amount our result is worse than the optimal system state: ", max_possible_value - our_loss, "\n");
	
		}

	#else

		device_do_schedule(N - 1, maxCPU, NUMGPUS, host_current_modes, d_losses, d_final_loss, host_uncooperative, d_final_solution, slack_A, slack_B, 0);

		loss = *d_final_loss;

	#endif

	std::vector<int> result;

	for (int i = 0; i < (int) num_tasks; i++)
		result.push_back(-1);

	for (int i = 0; i < (int) num_tasks; i++) {

		int index = real_order_of_tasks[i];
		

		#ifdef __NVCC__

			if (host_final[i] != -1)
				result.at(index) = (host_final[i]);

		#else

			if (d_final_solution[i] != -1)
				result.at(index) = (d_final_solution[i]);

		#endif

	}

	//determine ellapsed time in nanoseconds
	clock_gettime(CLOCK_MONOTONIC, &end_time);

	elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1e9;
	elapsed_time += (end_time.tv_nsec - start_time.tv_nsec);

	//print out the time taken
	print_module::print(std::cerr, "Time taken to run just the double knapsack: ", elapsed_time / 1000000, " milliseconds.\n");

	//check to see that we got a solution that renders this system schedulable
	if ((result.size() == 0 || loss == 100001) && first_time){

		print_module::print(std::cerr, "Error: System is not schedulable in any configuration. Exiting.\n");
		killpg(process_group, SIGUSR1);
		return;

	}
	
	else if ((result.size() == 0 || loss == 100001)){


		print_module::print(std::cerr, "Error: System is not schedulable in any configuration with specified constraints. Not updating modes.\n");

		for (int i = 0; i < schedule.count(); i++){
			(schedule.get_task(i))->reset_mode_to_previous();
			(schedule.get_task(i))->set_mode_transition(false);
		}

		return;

	}

	//make sure that all tasks which demanded
	//to be in a specific mode are in that mode
	//since we are guaranteed to have a valid
	//solution at this point
	for (int i = 0; i < schedule.count(); i++){

		if (!(schedule.get_task(i))->get_changeable() || !(schedule.get_task(i))->cooperative()){

			//if the task is not in the mode it was supposed to be in
			if ((schedule.get_task(i))->get_real_mode(result.at(i)) != (schedule.get_task(i))->get_real_current_mode()){

				print_module::print(std::cerr, "Error: Task ", i, " is not in the mode it was supposed to be in. Expected: ", (schedule.get_task(i))->get_real_mode(result.at(i)), " Found: ", (schedule.get_task(i))->get_real_current_mode(), "\n");
				killpg(process_group, SIGINT);
				return;

			}

		}

	}

	//print the new schedule layout
	std::ostringstream mode_strings;
	print_module::buffered_print(mode_strings, "\n========================= \n", "New Schedule Layout (virtual/real):\n");
	for (size_t i = 0; i < result.size(); i++)
		print_module::buffered_print(mode_strings, "Task ", i, " is now in mode: (", result.at(i), "/", schedule.get_task(i)->get_real_mode(result.at(i)), ")\n");
	print_module::buffered_print(mode_strings, "Total Loss from Mode Change: ", loss, "\n=========================\n\n");

	//print resources now held by each task
	print_module::buffered_print(mode_strings, "\n========================= \n", "New Resource Layout:\n");
	for (size_t i = 0; i < result.size(); i++)
		print_module::buffered_print(mode_strings, "Task ", i, " now has: ", task_table.at(i).at(result.at(i)).cores, " Core A | ", task_table.at(i).at(result.at(i)).sms, " Core B\n");
	print_module::buffered_print(mode_strings, "=========================\n\n");
	print_module::flush(std::cerr, mode_strings);

	//this changes the mode the tasks are currently
	//set to within their TaskData structure
	for (int i = 0; i < schedule.count(); i++)
		(schedule.get_task(i))->set_current_virtual_mode(result.at(i), false);

	//greedily give cpus on first run
	if (first_time) {
		
		//there is a variable used to store what the previous
		//mode that was active on the processor is in the TaskData
		//structure. Running this function for each task one more time
		//right at the start ensures that it is set
		for (int i = 0; i < schedule.count(); i++)
			(schedule.get_task(i))->set_current_virtual_mode(result.at(i), false);

		//update the previous modes to the first ever selected modes
		for (size_t i = 0; i < result.size(); i++)
			previous_modes.at(i) = (task_table.at(i).at(result.at(i)));

		int next_CPU = 1;

		//Actually assign CPUs to tasks. Start with 1.
		for (int i = 0; i < schedule.count(); i++){

			if ((schedule.get_task(i))->get_current_lowest_CPU() > 0){

				print_module::print(std::cerr, "Error in task ", i, ": all tasks should have had lowest CPU cleared. (this likely means memory was not cleaned up)\n");
				killpg(process_group, SIGINT);
				return;

			}

			(schedule.get_task(i))->set_current_lowest_CPU(next_CPU);
			next_CPU += (schedule.get_task(i))->get_current_CPUs();

			if (next_CPU > num_CPUs + 1){

				print_module::print(std::cerr, "Error in task ", i, ": too many CPUs have been allocated.", next_CPU, " ", num_CPUs, " \n");
				killpg(process_group, SIGINT);
				return;

			}		

		}

		//assign all the unassigned cpus to the scheduler to hold
		for (int i = next_CPU; i < num_CPUs; i++)
			schedule.get_task(result.size())->push_back_cpu(i);

		//Now assign TPC units to tasks, same method as before
		//(don't worry about holding TPC 1) 
		int next_TPC = 0;

		for (int i = 0; i < schedule.count(); i++){

			if ((schedule.get_task(i))->get_current_lowest_GPU() > 0){

				print_module::print(std::cerr, "Error in task ", i, ": all tasks should have had lowest GPU cleared. (this likely means memory was not cleaned up)\n");
				killpg(process_group, SIGINT);
				return;

			}

			//if this task actually has any TPCs assigned
			if (!(schedule.get_task(i))->pure_cpu_task()){

				(schedule.get_task(i))->set_current_lowest_GPU(next_TPC);

				for (int j = 0; j < (schedule.get_task(i))->get_current_GPUs(); j++)
					(schedule.get_task(i))->push_back_gpu(next_TPC ++);

				if (next_TPC > (int)(NUMGPUS) + 1){

					print_module::print(std::cerr, "Error in task ", i, ": too many GPUs have been allocated.", next_TPC, " ", NUMGPUS, " \n");
					killpg(process_group, SIGINT);
					return;

				}

			}
		}

		//assign all the unassigned gpus to the scheduler to hold
		for (int i = next_TPC; i < (int)(NUMGPUS); i++)
			schedule.get_task(result.size())->push_back_gpu(i);

	}

	//If we are not in the first execution, we cannot
	//do a greedy assignment of resources. We need to
	//build a resource allocation graph and execute it
	//only if the graph has no cycles
	else {

		//for each mode in result, subtract the new mode from the 
		//old mode to determine how many resources are being given up 
		//or taken from each task. This will be used to build the RAG.
		std::unordered_map<int, Node> nodes;
		std::unordered_map<int, Node> static_nodes;
		std::vector<std::pair<int, int>> dependencies;

		for (size_t i = 0; i < result.size(); i++){

			//fetch the current mode
			auto current_mode = task_table.at(i).at(result.at(i));

			//fetch the previous mode
			auto previous_mode = previous_modes.at(i);

			//add the new node
			dependencies.push_back({previous_mode.cores - current_mode.cores, previous_mode.sms - current_mode.sms});

		}

		//fetch all the resources we are supposed to be putting back into the 
		//free pool from queue 3

		//for all the free cores of both types, add them to the RAG
		//via adding a node that gives up that many resources
		dependencies.push_back({std::bitset<128>(schedule.get_task(result.size())->get_cpu_mask()).count(), std::bitset<128>(schedule.get_task(result.size())->get_gpu_mask()).count()});

		//build the copy of the results vector that the build resource graph function
		//needs as well as allocating an equal size vector to hold the lowest modes
		std::vector<int> task_modes = result;

		//for now just determine which modes dominate the others
		//for each task and build up the lowest_modes vector here.
		//FIXME: REPLACE LATER WITH CONSTANT ARRAY
		std::vector<int> lowest_modes(result.size(), 0);

		for (size_t i = 0; i < result.size(); i++){

			for (size_t j = 0; j < task_table.at(i).size(); j++){

				if (task_table.at(i).at(j).cores <= task_table.at(i).at(lowest_modes.at(i)).cores && task_table.at(i).at(j).sms <= task_table.at(i).at(lowest_modes.at(i)).sms)
					lowest_modes.at(i) = j;

			}

		}

		//if the call to build_resource_graph returns false, 
		//then we have a cycle and only a barrier can allow the handoff
		if (build_resource_graph(dependencies, nodes, static_nodes, task_modes, lowest_modes)){

			//show the resource graph (debugging)
			print_module::print(std::cerr, "\n========================= \n", "New Schedule RAG:\n");
			print_graph(nodes, static_nodes);
			print_module::print(std::cerr, "========================= \n\n");

			//by this point the RAG has either the previous solution inside of it, or it has
			//the current solution. Either way, we need to update the previous modes to reflect
			//the current modes.
			for (size_t i = 0; i < result.size(); i++){

				(schedule.get_task(i))->clear_cpus_granted_from_other_tasks();
				(schedule.get_task(i))->clear_gpus_granted_from_other_tasks();

			}

			//we now need to determine whether or not the RAG builder ended up building 
			//a single transition mode change or multiple mode changes. If it built a single
			//mode change, we can just execute the RAG. If it built multiple mode changes, we
			//need to do the first transition and then bring the system back up to the state 
			//indicated by the original result vector.
			bool multiple_mode_changes = false;
			for (int i = 0; i < (int)task_modes.size(); i++)
				if (task_modes.at(i) != result.at(i))
					multiple_mode_changes = true;

			//execute the RAG we proved exists
			execute_resource_allocation_graph(dependencies, nodes);

			//if we have a second mode change to bring the system back to the original state
			if (multiple_mode_changes){

				print_module::print(std::cerr, "Multiple Mode Changes Detected\n");

				//update the previous modes to the current modes
				for (size_t i = 0; i < result.size(); i++){

					previous_modes.at(i) = task_table.at(i).at(result.at(i));

					//notify all tasks that they should now transition
					(schedule.get_task(i))->set_mode_transition(false);

				}

				//signal the child processes
				pid_t process_group = getpgrp();
				killpg(process_group, SIGRTMIN+1);

				//wait for all tasks to finish transitioning
				for (int i = 0; i < schedule.count(); i++)
					while (!(schedule.get_task(i))->check_mode_transition());

				//now we need to rebuild the dependencies vector
				dependencies.clear();

				for (size_t i = 0; i < result.size(); i++){

					//fetch the current mode
					auto current_mode = task_table.at(i).at(result.at(i));

					//fetch the previous mode
					auto previous_mode = previous_modes.at(i);

					//add the new node
					dependencies.push_back({previous_mode.cores - current_mode.cores, previous_mode.sms - current_mode.sms});

				}

				//if we have multiple mode changes, we need to bring the system back to the original state
				//should be safe to do this without checking since we are just bringing providers and consumers
				//back up now????
				if (!build_resource_graph(dependencies, nodes, static_nodes, result, lowest_modes)){

					print_module::print(std::cerr, "Unexpected error: resource graph could not be built after multiple mode changes.\n");
					return;

				}

				//execute the RAG we proved exists
				execute_resource_allocation_graph(dependencies, nodes);

			}

		}	

		//This will be the position in which we fall back to multiple
		//mode changes to achieve what we want (as far as we can without
		//becoming 3 Partition that is)
		else{

			print_module::print(std::cerr, "Error: System was passed a RAG to build a DAG with, but a solution could not be found... skipping.\n");

			for (int i = 0; i < schedule.count(); i++){
				(schedule.get_task(i))->reset_mode_to_previous();
				(schedule.get_task(i))->set_mode_transition(false);
			}

			return;
	
		}

	}


	//update the previous modes to the current modes
	for (size_t i = 0; i < result.size(); i++){

		previous_modes.at(i) = task_table.at(i).at(result.at(i));

		//notify all tasks that they should now transition
		(schedule.get_task(i))->set_mode_transition(false);

	}

	first_time = false;

	clock_gettime(CLOCK_MONOTONIC, &end_time);

	//determine ellapsed time in milliseconds
	elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1e9;
	elapsed_time += (end_time.tv_nsec - start_time.tv_nsec);

	//print out the time taken
	print_module::print(std::cerr, "Time taken to reschedule: ", elapsed_time / 1000000, " milliseconds.\n");

}

void Scheduler::setTermination(){
	schedule.setTermination();
}
