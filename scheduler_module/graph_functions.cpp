/*************************************************************

The RAG builder needs its own copy of the simple 0-1 
knapsack so that the RAG can be built back since simple
greedy building will not allow a solution to be found
every time

**************************************************************/
void Scheduler::graph_building_zero_one_knapsack(int* out_array, int num_items, std::vector<int> transformers, std::unordered_map<int, Node>& nodes, int* max_weight, int* max_value, int side) {


	int dp_table[MAXTASKS + 1][128 + 1];
	for (int i = 0; i <= *max_weight; i++)
		dp_table[0][i] = 0;

	//look through the transformers and find the weights and values
	//for the side that we are currently working on
	std::vector<int> item_weights;
	std::vector<int> item_values;
	for (int i = 0; i < num_items; i++) {

		if (transformers[i] == -1) {
			item_weights.push_back(0);
			item_values.push_back(0);
			continue;
		}

		auto current_node = nodes[transformers[i]];

		if (current_node.x < 0){

			item_weights.push_back(current_node.x);
			item_values.push_back(current_node.y);

		}

		else {

			item_weights.push_back(current_node.y);
			item_values.push_back(current_node.x);

		}

	}

	std::cout << "Current Mode: " << ((side == 0) ? "A" : "B") << std::endl;

	//print all the weights and values
	for (int i = 0; i < num_items; i++) {

		if (transformers[i] == -1) continue;

		std::cout << "Item " << i << ": Weight: " << item_weights[i] << " Value: " << item_values[i] << std::endl;

	}

	std::cout << "Starting Knapsack" << std::endl;

	for (int i = 1; i <= num_items; i++) {

		for (int w = 0; w <= *max_weight; w++) {
		    
		    if (transformers.at(i - 1) == -1){
    		    dp_table[i][w] = dp_table[(i - 1)][w];
    		    continue;
    		}

			//skip any items which are not the ones
			//we are currently working on
			if ((side == 0 && nodes[transformers[i - 1]].x > 0) || (side == 1 && nodes[transformers[i - 1]].y > 0)){
			    dp_table[i][w] = dp_table[(i - 1)][w];
			    continue;
			}

			std::cout << "Item: " << i - 1 << " Weight: " << item_weights[i - 1] << " Value: " << item_values[i - 1] << " Max Weight: " << *max_weight << std::endl;

			dp_table[i][w] = dp_table[i - 1][w];

			if (-item_weights[i - 1] <= w){

				int old_value = dp_table[(i - 1)][w];
				int new_value = dp_table[(i - 1)][w + item_weights[i - 1]] + item_values[i - 1];

				dp_table[i][w] = (new_value > old_value) ? new_value : old_value;

			}
			
			else
				dp_table[i][w] = dp_table[(i - 1)][w];

		}

	}

	//backtrack to find the items
	int w = *max_weight;
	for (int i = num_items; i > 0; i--) {

		if (dp_table[i][w] != dp_table[i - 1][w]) {

			out_array[i - 1] = 1;
			w += item_weights[i - 1];

			//reduce the total weight for next time
			*max_weight += item_weights[i - 1];
			*max_value += item_values[i - 1];

		} 
		
		else {

			out_array[i - 1] = 0;

		}

	}

	//print the solution array
	std::cout << "Solution Array: ";

	for (int i = 0; i < num_items; i++) {

		std::cout << out_array[i] << " ";

	}

	std::cout << std::endl;

}

/*************************************************************

The problem itself is NP complete, but only for
specialized cases. The problem is often times very easy
to solve, and (ignoring combinatorial explosion due to 
order of transformers) can be solved in the same way
that we build the solution via the cautious scheduler.

If we cannot find a solution through this method, then 
we have to use multiple mode changes to achieve it.

*************************************************************/
bool Scheduler::build_resource_graph(std::vector<std::pair<int, int>> resource_pairs, 
                        std::unordered_map<int, Node>& nodes, std::unordered_map<int, Node>& static_nodes, std::vector<int>& task_modes,
						std::vector<int> lowest_modes) {
    nodes.clear();
    
    //create all nodes
    for (size_t i = 0; i < resource_pairs.size(); i++) {

		auto [x, y] = resource_pairs[i];
		nodes[i] = Node{(int)i, x, y, {}};
		static_nodes[i] = nodes[i];
    
	}
    
    //Legacy code, but nice to see the counts
    int provider_order = 0;
    int consumer_order = 0;
	int transfer_order = 0;

	std::ostringstream mode_strings;

	print_module::buffered_print(mode_strings, "\n========================= \n", "Nodes Passed to RAG Builder:\n");
    
    for (const auto& [id, node] : nodes) {

		if (id == (int) nodes.size() - 1)
			print_module::buffered_print(mode_strings, "Free Node <", node.x, ",", node.y, ">\n");
	
		else
			print_module::buffered_print(mode_strings, "Node ", id, " <", node.x, ",", node.y, ">\n");

		if (node.x == 0 && node.y == 0)
			continue;
    
	    if (node.x >= 0 && node.y >= 0)
            provider_order += 1;
    
	    if (node.x <= 0 && node.y <= 0)
            consumer_order += 1;

		if ((node.x < 0 && node.y > 0) || (node.x > 0 && node.y < 0))
			transfer_order += 1;

    }

	print_module::buffered_print(mode_strings, "=========================\n\n");
	print_module::flush(std::cerr, mode_strings);

	print_module::print(std::cerr, "Provider size: ", provider_order, " Consumer size: ", consumer_order, " Transfer size: ", transfer_order, "\n");
    
	//if system has barrier, just do it lazily
	if (barrier){

		for (int consumer_id = 0; consumer_id < (int) nodes.size(); consumer_id++){

			Node& consumer = nodes[consumer_id];
			int needed_x = -consumer.x;
			int needed_y = -consumer.y;
			
			for (int provider_id = 0; provider_id < (int) nodes.size(); provider_id++) {

				if (provider_id == consumer_id) continue;
				
				Node& provider = nodes[provider_id];
				Edge new_edge{consumer_id, 0, 0};
				bool edge_needed = false;
				
				//Try to satisfy x resource need
				if (needed_x > 0 && provider.x > 0) {

					int transfer = std::min(needed_x, provider.x);
					new_edge.x_amount = transfer;
					needed_x -= transfer;
					edge_needed = true;

				}
				
				//Try to satisfy y resource need
				if (needed_y > 0 && provider.y > 0) {

					int transfer = std::min(needed_y, provider.y);
					new_edge.y_amount = transfer;
					needed_y -= transfer;
					edge_needed = true;

				}
				
				//If this edge would transfer resources, add it and check for cycles
				if (edge_needed) {
					
					provider.edges.push_back(new_edge);

					//Update provider's available resources
					provider.x -= new_edge.x_amount;
					provider.y -= new_edge.y_amount;
				

				}

			}

		}

		return true;
	}

    //Just build the RAG the same way we did when 
	//we discovered the solution
	std::vector<int> discovered_providers;
	std::vector<int> discovered_consumers;
	std::vector<int> discovered_transformers;
	std::vector<int> original_providers_and_consumers;

	//reserve
	discovered_providers.reserve(nodes.size());
	discovered_consumers.reserve(nodes.size());

	//add the free pool to the providers
	discovered_providers.push_back(nodes.size() - 1);

	//loop over all nodes and classify them
	for (int i = 0; i < (int) nodes.size() - 1; i++){

		Node& node = nodes[i];

		if (node.x == 0 && node.y == 0)
			discovered_consumers.push_back(i);

		//if a pure provider, add it to the list
		else if (node.x >= 0 && node.y >= 0){

			discovered_providers.push_back(i);

			std::cout << "Provider: " << i << std::endl;

		}

		//if it's a consumer, just add it to the discovered
		else if (node.x <= 0 && node.y <= 0){

			discovered_consumers.push_back(i);

			std::cout << "Consumer: " << i << std::endl;

		}

		//if it's a transformer, add it to the transformer list
		else if ((node.x < 0 && node.y > 0) || (node.y < 0 && node.x > 0)){

			discovered_transformers.push_back(i);

			std::cout << "Transformer: " << i << std::endl;

		}

	}

	//make note of which providers were the original providers and consumers
	original_providers_and_consumers.reserve(discovered_providers.size() + discovered_consumers.size());
	original_providers_and_consumers.insert(original_providers_and_consumers.end(), discovered_providers.begin(), discovered_providers.end());
	original_providers_and_consumers.insert(original_providers_and_consumers.end(), discovered_consumers.begin(), discovered_consumers.end());

	//loop and discover all nodes and fix transformers
	//(providers are just ignored)
	int processed_transformers = 0;
	int last_recorded_transformers = -1;

	while (processed_transformers < (int) discovered_transformers.size()){

		//if we have no forward progress, we have a cycle
		//However, we might be able to break the cycle
		//by forcing some tasks to go into a lower state.
		//We do this one at a time, checking each time to see
		//if we were able to resolve the cycle. If we can't
		//resolve the cycle, and we have no other tasks to test with,
		//then we return false
		if (last_recorded_transformers == processed_transformers){

			//if we can lower just one task, it might be enough to break the cycle
			//in the graph
			bool lowered_task = false;

			//try to lower the mode of one of the original
			//producers or one of the consumers

			//FIXME: ADD DIFFERENT GREEDY FIT METHODS TO TRY HERE
			//FIRST-FIT, BEST-FIT, WORST-FIT, ETC.
			for (int& original_provider_or_consumer : original_providers_and_consumers){

				return false;

				Node& node = nodes[original_provider_or_consumer];

				//if we can't lower the mode, then we can't use this task
				if (!schedule.get_task(original_provider_or_consumer)->cooperative() || !schedule.get_task(original_provider_or_consumer)->get_changeable() || (task_modes[original_provider_or_consumer] == lowest_modes[original_provider_or_consumer]))
					continue;

				else {

					//note that we lowered a task mode
					lowered_task = true;

					//actually set this task to it's lowest mode
					task_modes[original_provider_or_consumer] = lowest_modes[original_provider_or_consumer];

					//if the task is a consumer we need to add it to the discovered providers and remove it from the consumers
					if (node.x == 0 && node.y == 0){

						discovered_providers.push_back(original_provider_or_consumer);
						discovered_consumers.erase(std::remove(discovered_consumers.begin(), discovered_consumers.end(), original_provider_or_consumer), discovered_consumers.end());

					}

					//update the x and y values of the node
					//(we subtract the difference between the current mode and the lowest mode
					//to get the amount of resources we need to add back to the graph)
					node.x = static_nodes[original_provider_or_consumer].x - task_table[original_provider_or_consumer].at(lowest_modes[original_provider_or_consumer]).cores; 
					node.y = static_nodes[original_provider_or_consumer].y - task_table[original_provider_or_consumer].at(lowest_modes[original_provider_or_consumer]).sms;

					//only do one at a time
					break;

				}

			}

			if (!lowered_task)
				return false;

		}

		//reset
		last_recorded_transformers = processed_transformers;

		//try to resolve the transformers
		for (int& current_transformer : discovered_transformers){

			if (current_transformer == -1)
				continue;

			//Node& node = nodes[current_transformer];

			Node& consumer = nodes[current_transformer];
			int needed_x = -consumer.x;
			int needed_y = -consumer.y;

			int possible_x = 0;
			int possible_y = 0;

			//check if it's even possible to satisfy the transformer
			for (int& provider_id : discovered_providers) {

				Node& provider = nodes[provider_id];

				possible_x += provider.x;
				possible_y += provider.y;

			}

			//if not, then return
			if (needed_x > 0 && possible_x < needed_x)
				continue;

			else if (needed_y > 0 && possible_y < needed_y)
				continue;

			//otherwise satisfy the requirements
			for (int provider_id : discovered_providers) {
			
				Node& provider = nodes[provider_id];
				Edge new_edge{current_transformer, 0, 0};
				bool edge_needed = false;
				
				//Try to satisfy x resource need
				if (needed_x > 0 && provider.x > 0) {

					int transfer = std::min(needed_x, provider.x);
					new_edge.x_amount = transfer;
					needed_x -= transfer;
					edge_needed = true;

				}
				
				//Try to satisfy y resource need
				if (needed_y > 0 && provider.y > 0) {

					int transfer = std::min(needed_y, provider.y);
					new_edge.y_amount = transfer;
					needed_y -= transfer;
					edge_needed = true;

				}
				
				//If this edge would transfer resources, add it and check for cycles
				if (edge_needed) {
					
					provider.edges.push_back(new_edge);

					//Update provider's available resources
					provider.x -= new_edge.x_amount;
					provider.y -= new_edge.y_amount;

				}

			}

			//now this once transformer is a provider
			if (consumer.x < 0)
				consumer.x = 0;
			if (consumer.y < 0)
				consumer.y = 0;

			discovered_providers.push_back(current_transformer);

			current_transformer = -1;

			processed_transformers += 1;

		}

	}

	//now just do the same thing we did for transformers
	//but with the discovered consumers
	for (int consumer_id : discovered_consumers){

		Node& consumer = nodes[consumer_id];
		int needed_x = -consumer.x;
		int needed_y = -consumer.y;

		for (int provider_id : discovered_providers) {
		
			Node& provider = nodes[provider_id];
			Edge new_edge{consumer_id, 0, 0};
			bool edge_needed = false;
			
			//Try to satisfy x resource need
			if (needed_x > 0 && provider.x > 0) {

				int transfer = std::min(needed_x, provider.x);
				new_edge.x_amount = transfer;
				needed_x -= transfer;
				edge_needed = true;

			}
			
			//Try to satisfy y resource need
			if (needed_y > 0 && provider.y > 0) {

				int transfer = std::min(needed_y, provider.y);
				new_edge.y_amount = transfer;
				needed_y -= transfer;
				edge_needed = true;

			}
			
			//If this edge would transfer resources, add it and check for cycles
			if (edge_needed) {
				
				provider.edges.push_back(new_edge);

				//Update provider's available resources
				provider.x -= new_edge.x_amount;
				provider.y -= new_edge.y_amount;

			}

		}
		
	}

	std::cerr << "Graph built successfully" << std::endl;
	
    return true;

}

/*************************************************************

When we are constructing a resource allocation graph after
building up the solution and using a reordering, we will likely
(although not guaranteed) need to rebuild the graph using
the zero-one knapsack algorithm. This function acts exactly
as it's dumber predecessor, but it should successfully
rebuild where greedy building will likely fail

*************************************************************/
bool Scheduler::build_resource_graph_zero_one(std::vector<std::pair<int, int>> resource_pairs, 
                        std::unordered_map<int, Node>& nodes, std::unordered_map<int, Node>& static_nodes, std::vector<int>& task_modes,
						std::vector<int> lowest_modes, bool execute) {
    nodes.clear();

	//setup a vector to store the masks of the tasks
	//which are giving resources to other tasks
	std::vector<__uint128_t> task_masks;
    
    //create all nodes
    for (size_t i = 0; i < resource_pairs.size(); i++) {

		auto [x, y] = resource_pairs[i];
		nodes[i] = Node{(int)i, x, y, {}};
		static_nodes[i] = nodes[i];

		task_masks.push_back(0);
    
	}
    
    int provider_order = 0;
    int consumer_order = 0;
	int transfer_order = 0;

	int slack_A = 0;
	int slack_B = 0;

	std::ostringstream mode_strings;
    
    for (const auto& [id, node] : nodes) {

		if (node.x == 0 && node.y == 0)
			continue;
    
	    if (node.x >= 0 && node.y >= 0){

		    provider_order += 1;

		}
    
	    if (node.x <= 0 && node.y <= 0)
            consumer_order += 1;

		if ((node.x < 0 && node.y > 0) || (node.x > 0 && node.y < 0))
			transfer_order += 1;

    }

    //Just build the RAG the same way we did when 
	//we discovered the solution
	std::vector<int> discovered_providers;
	std::vector<int> discovered_consumers;
	std::vector<int> discovered_transformers;
	std::vector<int> original_providers_and_consumers;

	//reserve
	discovered_providers.reserve(nodes.size());
	discovered_consumers.reserve(nodes.size());

	//add the free pool to the providers
	discovered_providers.push_back(nodes.size() - 1);

	//loop over all nodes and classify them
	for (int i = 0; i <= (int) nodes.size() - 1; i++){

		Node& node = nodes[i];

		if (i == nodes.size() - 1){

			slack_A += node.x;
			slack_B += node.y;
			
			continue;

		}

		if (node.x == 0 && node.y == 0)
			discovered_consumers.push_back(i);

		//if a pure provider, add it to the list
		else if (node.x >= 0 && node.y >= 0){

			discovered_providers.push_back(i);

			slack_A += node.x;
			slack_B += node.y;

		}

		//if it's a consumer, just add it to the discovered
		else if (node.x <= 0 && node.y <= 0){

			discovered_consumers.push_back(i);

		}

		//if it's a transformer, add it to the transformer list
		else if ((node.x < 0 && node.y > 0) || (node.y < 0 && node.x > 0)){

			discovered_transformers.push_back(i);

		}

	}

	//make note of which providers were the original providers and consumers
	original_providers_and_consumers.reserve(discovered_providers.size() + discovered_consumers.size());
	original_providers_and_consumers.insert(original_providers_and_consumers.end(), discovered_providers.begin(), discovered_providers.end());
	original_providers_and_consumers.insert(original_providers_and_consumers.end(), discovered_consumers.begin(), discovered_consumers.end());

	//loop and discover all nodes and fix transformers
	//(providers are just ignored)
	int processed_transformers = 0;
	int last_recorded_transformers = -1;

	while (processed_transformers < (int) discovered_transformers.size()){

		//if we have no forward progress, we have a cycle
		//However, we might be able to break the cycle
		//by forcing some tasks to go into a lower state.
		//We do this one at a time, checking each time to see
		//if we were able to resolve the cycle. If we can't
		//resolve the cycle, and we have no other tasks to test with,
		//then we return false
		if (last_recorded_transformers == processed_transformers)
			return false;

		//reset
		last_recorded_transformers = processed_transformers;

		//run the zero one knapsack flipping between sides
		for (int i = 0; i <= 1; i++){

			//output array 
			int out_array[discovered_transformers.size()];

			//run the knapsack for our given side
			graph_building_zero_one_knapsack(out_array, discovered_transformers.size(), discovered_transformers, nodes, (i == 0) ? &slack_A : &slack_B, (i == 0) ? &slack_B : &slack_A, i);

			//try to resolve the transformers
			for (int i = 0; i < discovered_transformers.size(); i++){
			
			    int current_transformer = out_array[i];

				if (current_transformer == 0)
					continue;

				current_transformer = discovered_transformers[i];

				Node& consumer = nodes[current_transformer];
				int needed_x = -consumer.x;
				int needed_y = -consumer.y;

				int possible_x = 0;
				int possible_y = 0;

				//check if it's even possible to satisfy the transformer
				for (int& provider_id : discovered_providers) {

					Node& provider = nodes[provider_id];

					possible_x += provider.x;
					possible_y += provider.y;

				}

				//if not, then return
				if (needed_x > 0 && possible_x < needed_x)
					continue;

				else if (needed_y > 0 && possible_y < needed_y)
					continue;

				//otherwise satisfy the requirements
				for (int provider_id : discovered_providers) {
				
					Node& provider = nodes[provider_id];
					Edge new_edge{current_transformer, 0, 0};
					bool edge_needed = false;
					
					//Try to satisfy x resource need
					if (needed_x > 0 && provider.x > 0) {

						int transfer = std::min(needed_x, provider.x);
						new_edge.x_amount = transfer;
						needed_x -= transfer;
						edge_needed = true;

						//place message in queue for the task giving up processors
						if (execute){

							schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(current_transformer, 0, transfer);

							//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << current_transformer << std::endl;

							//store the mask
							task_masks[current_transformer] |= ((__uint128_t) 1 << provider_id);

						}

					}
					
					//Try to satisfy y resource need
					if (needed_y > 0 && provider.y > 0) {

						int transfer = std::min(needed_y, provider.y);
						new_edge.y_amount = transfer;
						needed_y -= transfer;
						edge_needed = true;

						//place message in queue for the task giving up processors
						if (execute){

							schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(current_transformer, 1, transfer);

							//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << current_transformer << std::endl;

							//store the mask
							task_masks[current_transformer] |= ((__uint128_t) 1 << provider_id);

						}

					}
					
					//If this edge would transfer resources, add it and check for cycles
					if (edge_needed) {
						
						provider.edges.push_back(new_edge);

						//Update provider's available resources
						provider.x -= new_edge.x_amount;
						provider.y -= new_edge.y_amount;

					}

				}

				//now this once transformer is a provider
				if (consumer.x < 0)
					consumer.x = 0;
				if (consumer.y < 0)
					consumer.y = 0;

				discovered_providers.push_back(current_transformer);

				discovered_transformers[i] = -1;

				processed_transformers += 1;

			}
		
		}

	}

	//now just do the same thing we did for transformers
	//but with the discovered consumers
	for (int consumer_id : discovered_consumers){

		Node& consumer = nodes[consumer_id];
		int needed_x = -consumer.x;
		int needed_y = -consumer.y;

		for (int provider_id : discovered_providers) {
		
			Node& provider = nodes[provider_id];
			Edge new_edge{consumer_id, 0, 0};
			bool edge_needed = false;
			
			//Try to satisfy x resource need
			if (needed_x > 0 && provider.x > 0) {

				int transfer = std::min(needed_x, provider.x);
				new_edge.x_amount = transfer;
				needed_x -= transfer;
				edge_needed = true;

				if (execute){

					//place message in queue for the task giving up processors
					schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 0, transfer);

					//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << consumer_id << std::endl;

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}

			}
			
			//Try to satisfy y resource need
			if (needed_y > 0 && provider.y > 0) {

				int transfer = std::min(needed_y, provider.y);
				new_edge.y_amount = transfer;
				needed_y -= transfer;
				edge_needed = true;

				if (execute){

					//place message in queue for the task giving up processors
					schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 1, transfer);

					//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << consumer_id << std::endl;

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}

			}
			
			//If this edge would transfer resources, add it and check for cycles
			if (edge_needed) {
				
				provider.edges.push_back(new_edge);

				//Update provider's available resources
				provider.x -= new_edge.x_amount;
				provider.y -= new_edge.y_amount;

			}

		}
		
	}

	if (execute){

		//for all the tasks which are providers and haven't 
		//transferred all their resources, we need to send 
		//all the additional resources to the free pool
		for (int i = 0; i < (int) discovered_providers.size(); i++){

			Node& provider = nodes[discovered_providers[i]];

			if (discovered_providers[i] == (int) nodes.size() - 1)
				continue;

			if (provider.x > 0){

				//place message in queue for the task giving up processors
				schedule.get_task(discovered_providers[i])->set_processors_to_send_to_other_processes(nodes.size() - 1, 0, provider.x);

				//std::cerr << "Sending " << provider.x << " processors from " << discovered_providers[i] << " to free pool" << std::endl;

			}

			if (provider.y > 0){

				//place message in queue for the task giving up processors
				schedule.get_task(discovered_providers[i])->set_processors_to_send_to_other_processes(nodes.size() - 1, 1, provider.y);

				//std::cerr << "Sending " << provider.y << " processors from " << discovered_providers[i] << " to free pool" << std::endl;

			}

		}


		//now we need to send messages to the tasks to tell them
		//what tasks they are waiting on to transition
		//std::cerr << "TOTAL TASK MASKS PRESENT: " << task_masks.size() << std::endl;
		for (int i = 0; i < (int) task_masks.size() - 1; i++)
			schedule.get_task(i)->set_tasks_to_wait_on(task_masks[i]);

		//finally read all the messages for the free pool and send 
		schedule.get_task(nodes.size() - 1)->give_processors_to_other_tasks();

		return true;
			
	}

	else {

		std::cerr << "Graph built successfully" << std::endl;

		//call self with execute true
		build_resource_graph_zero_one(resource_pairs, nodes, static_nodes, task_modes, lowest_modes, true);
		
		return true;

	}

}


/*************************************************************

If we verified that we can build a RAG, we just run the 
same algorithm, but this time we just send messages to the 
tasks via the message queues to engage the exchanges

*************************************************************/

//FIXME: THIS WILL ATTEMPT TO SEND MESSAGES THROUGH THE FREE POOL
//VIA IT'S INDEX WHICH WILL BE AN OUT OF BOUND READ. IF THE FREE POOL
//NEEDS TO BE INTERACTED WITH, WE NEED TO USE THE STORED REFERENCE, OR WE
//NEED TO ADD THE TASK DATA TO THE LIST, BUT THAT REQUIRES MAKING SURE
//WE DO NOT USE THE SIZE OF THE TASK DATA WHICH SEEMS PROBLEMATIC
void Scheduler::execute_resource_allocation_graph(std::vector<std::pair<int, int>> resource_pairs, 
                        std::unordered_map<int, Node>& nodes) {
    nodes.clear();

	//setup a vector to store the masks of the tasks
	//which are giving resources to other tasks
	std::vector<__uint128_t> task_masks;
    
    //create all nodes
    for (size_t i = 0; i < resource_pairs.size(); i++) {

		auto [x, y] = resource_pairs[i];
		nodes[i] = Node{(int)i, x, y, {}};

		task_masks.push_back(0);
    
	}

	//if system has barrier, just do it lazily
	if (barrier){

		for (int consumer_id = 0; consumer_id < (int) nodes.size(); consumer_id++){

			Node& consumer = nodes[consumer_id];
			int needed_x = -consumer.x;
			int needed_y = -consumer.y;
			
			for (int provider_id = 0; provider_id < (int) nodes.size(); provider_id++) {

				if (provider_id == consumer_id) continue;
				
				Node& provider = nodes[provider_id];
				Edge new_edge{consumer_id, 0, 0};
				
				//Try to satisfy x resource need
				if (needed_x > 0 && provider.x > 0) {

					int transfer = std::min(needed_x, provider.x);
					new_edge.x_amount = transfer;
					needed_x -= transfer;
					provider.x -= new_edge.x_amount;

					//place message in queue for the task giving up processors
					if (provider_id != (int) nodes.size() - 1)
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 0, transfer);

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}
				
				//Try to satisfy y resource need
				if (needed_y > 0 && provider.y > 0) {

					int transfer = std::min(needed_y, provider.y);
					new_edge.y_amount = transfer;
					needed_y -= transfer;
					provider.y -= new_edge.y_amount;
					

					//place message in queue for the task giving up processors
					if (provider_id != (int) nodes.size() - 1)
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 1, transfer);

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}

			}

		}

		//FIXME: BARRIER VERSION JUST NOT RIGHT AT ALL

	}

	else {

		//Just build the RAG the same way we did when 
		//we discovered the solution
		std::vector<int> discovered_providers;
		std::vector<int> discovered_consumers;
		std::vector<int> discovered_transformers;

		//reserve
		discovered_providers.reserve(nodes.size());
		discovered_consumers.reserve(nodes.size());

		//add the free pool to the providers
		discovered_providers.push_back(nodes.size() - 1);

		//loop over all nodes and classify them
		for (int i = 0; i < (int) nodes.size() - 1; i++){

			Node& node = nodes[i];

			if (node.x == 0 && node.y == 0)
				discovered_consumers.push_back(i);

			//if a pure provider, add it to the list
			else if (node.x >= 0 && node.y >= 0)
				discovered_providers.push_back(i);

			//if it's a consumer, just add it to the discovered
			else if (node.x <= 0 && node.y <= 0)
				discovered_consumers.push_back(i);


			//if it's a transformer, add it to the transformer list
			else if ((node.x < 0 && node.y > 0) || (node.y < 0 && node.x > 0))
				discovered_transformers.push_back(i);

		}

		//loop and discover all nodes and fix transformers
		//(providers are just ignored)
		int processed_transformers = 0;

		while (processed_transformers < (int) discovered_transformers.size()){

			//try to resolve the transformers
			for (int& current_transformer : discovered_transformers){

				if (current_transformer == -1)
					continue;

				//Node& node = nodes[current_transformer];

				Node& consumer = nodes[current_transformer];
				int needed_x = -consumer.x;
				int needed_y = -consumer.y;

				int possible_x = 0;
				int possible_y = 0;

				//check if it's even possible to satisfy the transformer
				for (int& provider_id : discovered_providers) {

					Node& provider = nodes[provider_id];

					possible_x += provider.x;
					possible_y += provider.y;

				}

				//if not, then return
				if (needed_x > 0 && possible_x < needed_x)
					continue;

				else if (needed_y > 0 && possible_y < needed_y)
					continue;

				//otherwise satisfy the requirements
				for (int provider_id : discovered_providers) {
				
					Node& provider = nodes[provider_id];
					Edge new_edge{current_transformer, 0, 0};
					
					//Try to satisfy x resource need
					if (needed_x > 0 && provider.x > 0) {

						int transfer = std::min(needed_x, provider.x);
						new_edge.x_amount = transfer;
						needed_x -= transfer;
						provider.x -= new_edge.x_amount;

						//place message in queue for the task giving up processors
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(current_transformer, 0, transfer);

						//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << current_transformer << std::endl;

						//store the mask
						task_masks[current_transformer] |= ((__uint128_t) 1 << provider_id);

					}
					
					//Try to satisfy y resource need
					if (needed_y > 0 && provider.y > 0) {

						int transfer = std::min(needed_y, provider.y);
						new_edge.y_amount = transfer;
						needed_y -= transfer;
						provider.y -= new_edge.y_amount;

						//place message in queue for the task giving up processors
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(current_transformer, 1, transfer);

						//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << current_transformer << std::endl;

						//store the mask
						task_masks[current_transformer] |= ((__uint128_t) 1 << provider_id);

					}

				}

				//now this once transformer is a provider
				if (consumer.x < 0)
					consumer.x = 0;
				if (consumer.y < 0)
					consumer.y = 0;

				discovered_providers.push_back(current_transformer);

				current_transformer = -1;

				processed_transformers += 1;

			}

		}

		//now just do the same thing we did for transformers
		//but with the discovered consumers
		for (int consumer_id : discovered_consumers){

			Node& consumer = nodes[consumer_id];
			int needed_x = -consumer.x;
			int needed_y = -consumer.y;

			for (int provider_id : discovered_providers) {
			
				Node& provider = nodes[provider_id];
				Edge new_edge{consumer_id, 0, 0};
				
				//Try to satisfy x resource need
				if (needed_x > 0 && provider.x > 0) {

					int transfer = std::min(needed_x, provider.x);
					new_edge.x_amount = transfer;
					needed_x -= transfer;
					provider.x -= new_edge.x_amount;

					//place message in queue for the task giving up processors
					schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 0, transfer);

					//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << consumer_id << std::endl;

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}
				
				//Try to satisfy y resource need
				if (needed_y > 0 && provider.y > 0) {

					int transfer = std::min(needed_y, provider.y);
					new_edge.y_amount = transfer;
					needed_y -= transfer;
					provider.y -= new_edge.y_amount;

					//place message in queue for the task giving up processors
					schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 1, transfer);

					//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << consumer_id << std::endl;

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}

			}
			
		}

		//for all the tasks which are providers and haven't 
		//transferred all their resources, we need to send 
		//all the additional resources to the free pool
		for (int i = 0; i < (int) discovered_providers.size(); i++){

			Node& provider = nodes[discovered_providers[i]];

			if (discovered_providers[i] == (int) nodes.size() - 1)
				continue;

			if (provider.x > 0){

				//place message in queue for the task giving up processors
				schedule.get_task(discovered_providers[i])->set_processors_to_send_to_other_processes(nodes.size() - 1, 0, provider.x);

				//std::cerr << "Sending " << provider.x << " processors from " << discovered_providers[i] << " to free pool" << std::endl;

			}

			if (provider.y > 0){

				//place message in queue for the task giving up processors
				schedule.get_task(discovered_providers[i])->set_processors_to_send_to_other_processes(nodes.size() - 1, 1, provider.y);

				//std::cerr << "Sending " << provider.y << " processors from " << discovered_providers[i] << " to free pool" << std::endl;

			}

		}

	}


	//now we need to send messages to the tasks to tell them
	//what tasks they are waiting on to transition
	//std::cerr << "TOTAL TASK MASKS PRESENT: " << task_masks.size() << std::endl;
	for (int i = 0; i < (int) task_masks.size() - 1; i++)
		schedule.get_task(i)->set_tasks_to_wait_on(task_masks[i]);

	//finally read all the messages for the free pool and send 
	schedule.get_task(nodes.size() - 1)->give_processors_to_other_tasks();
}


//convert the print_graph function to use buffered print
void Scheduler::print_graph(const std::unordered_map<int, Node>& nodes, std::unordered_map<int, Node> static_nodes) {

	std::ostringstream mode_strings;

	print_module::buffered_print(mode_strings, "\nNodes and resource transfers:\n");
	
	for (const auto& [id, node] : nodes) {

		if (id != ((int) nodes.size() - 1))
			print_module::buffered_print(mode_strings, "Node ", id, " <", static_nodes[id].x, ",", static_nodes[id].y, "> → ");
		else
			print_module::buffered_print(mode_strings, "Free Resources", " <", static_nodes[id].x, ",", static_nodes[id].y, "> → ");

		if (node.edges.empty())
			print_module::buffered_print(mode_strings, "no edges");

		else {
			
			for (const Edge& edge : node.edges) {

				print_module::buffered_print(mode_strings, edge.to_node, "(");
				bool first = true;

				if (edge.x_amount > 0) {
				
				    print_module::buffered_print(mode_strings, "x:", edge.x_amount);
					first = false;
				
				}

				if (edge.y_amount > 0) {
				
				    if (!first) print_module::buffered_print(mode_strings, ",");
					print_module::buffered_print(mode_strings, "y:", edge.y_amount);
				
				}
				
				print_module::buffered_print(mode_strings, ") ");
			}
		}
		print_module::buffered_print(mode_strings, "\n");
	}
	print_module::buffered_print(mode_strings, "\n");

	print_module::flush(std::cerr, mode_strings);
}