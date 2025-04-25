/*************************************************************

The problem itself is possibly NP complete, but only for
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

			return false;

			//if we can lower just one task, it might be enough to break the cycle
			//in the graph
			bool lowered_task = false;

			//try to lower the mode of one of the original
			//producers or one of the consumers

			//FIXME: ADD DIFFERENT GREEDY FIT METHODS TO TRY HERE
			//FIRST-FIT, BEST-FIT, WORST-FIT, ETC.
			for (int& original_provider_or_consumer : original_providers_and_consumers){

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

			Node& transformer = nodes[current_transformer];

			int needed_x = -transformer.x;
			int needed_y = -transformer.y;

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
			if (transformer.x < 0)
				transformer.x = 0;
			if (transformer.y < 0)
				transformer.y = 0;

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

				Node& transformer = nodes[current_transformer];

				int needed_x = -transformer.x;
				int needed_y = -transformer.y;

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
				if (transformer.x < 0)
					transformer.x = 0;
				if (transformer.y < 0)
					transformer.y = 0;

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
	for (int i = 0; i < (int) task_masks.size(); i++)
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