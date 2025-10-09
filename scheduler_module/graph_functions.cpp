/*************************************************************

The problem itself is possibly NP complete, but only for
specialized cases. The problem is often times very easy
to solve, and (ignoring combinatorial explosion due to 
order of transformers) can be solved in the same way
that we build the solution via the cautious scheduler.

If we cannot find a solution through this method, then 
we have to use multiple mode changes to achieve it.

*************************************************************/
bool Scheduler::build_resource_graph(std::vector<std::tuple<int, int, int, int>> resource_pairs, 
                        std::unordered_map<int, Node>& nodes, std::unordered_map<int, Node>& static_nodes, std::vector<int>& task_modes,
						std::vector<int> lowest_modes) {
    nodes.clear();
    
    //create all nodes
    for (size_t i = 0; i < resource_pairs.size(); i++) {

		auto [x, y, z, w] = resource_pairs[i];
		nodes[i] = Node{(int)i, x, y, z, w, {}};
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
			print_module::buffered_print(mode_strings, "Free Node <", node.a, ",", node.b, ",", node.c, ",", node.d, ">\n");
	
		else
			print_module::buffered_print(mode_strings, "Node ", id, " <", node.a, ",", node.b, ",", node.c, ",", node.d, ">\n");

		if (node.a == 0 && node.b == 0 && node.c == 0 && node.d == 0)
			continue;
    
	    if (node.a >= 0 && node.b >= 0 && node.c >= 0 && node.d >= 0)
            provider_order += 1;
    
	    if (node.a <= 0 && node.b <= 0 && node.c <= 0 && node.d <= 0)
            consumer_order += 1;

		if ((node.a < 0 && node.b > 0) || (node.a > 0 && node.b < 0) || (node.c < 0 && node.d > 0) || (node.c > 0 && node.d < 0) || (node.a < 0 && node.c > 0) || (node.a > 0 && node.c < 0) || (node.a < 0 && node.d > 0) || (node.a > 0 && node.d < 0) || (node.b < 0 && node.c > 0) || (node.b > 0 && node.c < 0) || (node.b < 0 && node.d > 0) || (node.b > 0 && node.d < 0))
			transfer_order += 1;

    }

	print_module::buffered_print(mode_strings, "=========================\n\n");
	print_module::flush(std::cerr, mode_strings);

	print_module::print(std::cerr, "Provider size: ", provider_order, " Consumer size: ", consumer_order, " Transfer size: ", transfer_order, "\n");
    
	//if system has barrier, just do it lazily
	if (barrier){

		for (int consumer_id = 0; consumer_id < (int) nodes.size(); consumer_id++){

			Node& consumer = nodes[consumer_id];
			int needed_a = -consumer.a;
			int needed_b = -consumer.b;
			int needed_c = -consumer.c;
			int needed_d = -consumer.d;
			
			for (int provider_id = 0; provider_id < (int) nodes.size(); provider_id++) {

				if (provider_id == consumer_id) continue;
				
				Node& provider = nodes[provider_id];
				Edge new_edge{consumer_id, 0, 0, 0, 0};
				bool edge_needed = false;
				
				//Try to satisfy a resource need
				if (needed_a > 0 && provider.a > 0) {

					int transfer = std::min(needed_a, provider.a);
					new_edge.a_amount = transfer;
					needed_a -= transfer;
					edge_needed = true;

				}
				
				//Try to satisfy b resource need
				if (needed_b > 0 && provider.b > 0) {

					int transfer = std::min(needed_b, provider.b);
					new_edge.b_amount = transfer;
					needed_b -= transfer;
					edge_needed = true;

				}
				
				//Try to satisfy c resource need
				if (needed_c > 0 && provider.c > 0) {

					int transfer = std::min(needed_c, provider.c);
					new_edge.c_amount = transfer;
					needed_c -= transfer;
					edge_needed = true;

				}
				
				//Try to satisfy d resource need
				if (needed_d > 0 && provider.d > 0) {

					int transfer = std::min(needed_d, provider.d);
					new_edge.d_amount = transfer;
					needed_d -= transfer;
					edge_needed = true;

				}
				
				//If this edge would transfer resources, add it and check for cycles
				if (edge_needed) {
					
					provider.edges.push_back(new_edge);

					//Update provider's available resources
					provider.a -= new_edge.a_amount;
					provider.b -= new_edge.b_amount;
					provider.c -= new_edge.c_amount;
					provider.d -= new_edge.d_amount;
				

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

		if (node.a == 0 && node.b == 0 && node.c == 0 && node.d == 0)
			discovered_consumers.push_back(i);

		//if a pure provider, add it to the list
		else if (node.a >= 0 && node.b >= 0 && node.c >= 0 && node.d >= 0){

			discovered_providers.push_back(i);

			std::cout << "Provider: " << i << std::endl;

		}

		//if it's a consumer, just add it to the discovered
		else if (node.a <= 0 && node.b <= 0 && node.c <= 0 && node.d <= 0){

			discovered_consumers.push_back(i);

			std::cout << "Consumer: " << i << std::endl;

		}

		//if it's a transformer, add it to the transformer list
		else if ((node.a < 0 && node.b > 0) || (node.b < 0 && node.a > 0) || (node.c < 0 && node.d > 0) || (node.d < 0 && node.c > 0) || (node.a < 0 && node.c > 0) || (node.c < 0 && node.a > 0) || (node.a < 0 && node.d > 0) || (node.d < 0 && node.a > 0) || (node.b < 0 && node.c > 0) || (node.c < 0 && node.b > 0) || (node.b < 0 && node.d > 0) || (node.d < 0 && node.b > 0)){

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

		//reset
		last_recorded_transformers = processed_transformers;

		//try to resolve the transformers
		for (int& current_transformer : discovered_transformers){

			if (current_transformer == -1)
				continue;

			//Node& node = nodes[current_transformer];

			Node& consumer = nodes[current_transformer];
			int needed_a = -consumer.a;
			int needed_b = -consumer.b;
			int needed_c = -consumer.c;
			int needed_d = -consumer.d;

			int possible_a = 0;
			int possible_b = 0;
			int possible_c = 0;
			int possible_d = 0;

			//check if it's even possible to satisfy the transformer
			for (int& provider_id : discovered_providers) {

				Node& provider = nodes[provider_id];

				possible_a += provider.a;
				possible_b += provider.b;
				possible_c += provider.c;
				possible_d += provider.d;

			}

			//if not, then return
			if (needed_a > 0 && possible_a < needed_a)
				continue;

			else if (needed_b > 0 && possible_b < needed_b)
				continue;

			else if (needed_c > 0 && possible_c < needed_c)
				continue;

			else if (needed_d > 0 && possible_d < needed_d)
				continue;

			//otherwise satisfy the requirements
			for (int provider_id : discovered_providers) {
			
				Node& provider = nodes[provider_id];
				Edge new_edge{current_transformer, 0, 0, 0, 0};
				bool edge_needed = false;
				
				//Try to satisfy a resource need
				if (needed_a > 0 && provider.a > 0) {

					int transfer = std::min(needed_a, provider.a);
					new_edge.a_amount = transfer;
					needed_a -= transfer;
					edge_needed = true;

				}
				
				//Try to satisfy b resource need
				if (needed_b > 0 && provider.b > 0) {

					int transfer = std::min(needed_b, provider.b);
					new_edge.b_amount = transfer;
					needed_b -= transfer;
					edge_needed = true;

				}
				
				//Try to satisfy c resource need
				if (needed_c > 0 && provider.c > 0) {

					int transfer = std::min(needed_c, provider.c);
					new_edge.c_amount = transfer;
					needed_c -= transfer;
					edge_needed = true;

				}
				
				//Try to satisfy d resource need
				if (needed_d > 0 && provider.d > 0) {

					int transfer = std::min(needed_d, provider.d);
					new_edge.d_amount = transfer;
					needed_d -= transfer;
					edge_needed = true;

				}
				
				//If this edge would transfer resources, add it and check for cycles
				if (edge_needed) {
					
					provider.edges.push_back(new_edge);

					//Update provider's available resources
					provider.a -= new_edge.a_amount;
					provider.b -= new_edge.b_amount;
					provider.c -= new_edge.c_amount;
					provider.d -= new_edge.d_amount;

				}

			}

			//now this once transformer is a provider
			if (consumer.a < 0)
				consumer.a = 0;
			if (consumer.b < 0)
				consumer.b = 0;
			if (consumer.c < 0)
				consumer.c = 0;
			if (consumer.d < 0)
				consumer.d = 0;

			discovered_providers.push_back(current_transformer);

			current_transformer = -1;

			processed_transformers += 1;

		}

	}

	//now just do the same thing we did for transformers
	//but with the discovered consumers
	for (int consumer_id : discovered_consumers){

		Node& consumer = nodes[consumer_id];
		int needed_a = -consumer.a;
		int needed_b = -consumer.b;
		int needed_c = -consumer.c;
		int needed_d = -consumer.d;

		for (int provider_id : discovered_providers) {
		
			Node& provider = nodes[provider_id];
			Edge new_edge{consumer_id, 0, 0, 0, 0};
			bool edge_needed = false;
			
			//Try to satisfy a resource need
			if (needed_a > 0 && provider.a > 0) {

				int transfer = std::min(needed_a, provider.a);
				new_edge.a_amount = transfer;
				needed_a -= transfer;
				edge_needed = true;

			}
			
			//Try to satisfy b resource need
			if (needed_b > 0 && provider.b > 0) {

				int transfer = std::min(needed_b, provider.b);
				new_edge.b_amount = transfer;
				needed_b -= transfer;
				edge_needed = true;

			}
			
			//Try to satisfy c resource need
			if (needed_c > 0 && provider.c > 0) {

				int transfer = std::min(needed_c, provider.c);
				new_edge.c_amount = transfer;
				needed_c -= transfer;
				edge_needed = true;

			}
			
			//Try to satisfy d resource need
			if (needed_d > 0 && provider.d > 0) {

				int transfer = std::min(needed_d, provider.d);
				new_edge.d_amount = transfer;
				needed_d -= transfer;
				edge_needed = true;

			}
			
			//If this edge would transfer resources, add it and check for cycles
			if (edge_needed) {
				
				provider.edges.push_back(new_edge);

				//Update provider's available resources
				provider.a -= new_edge.a_amount;
				provider.b -= new_edge.b_amount;
				provider.c -= new_edge.c_amount;
				provider.d -= new_edge.d_amount;

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
void Scheduler::execute_resource_allocation_graph(std::vector<std::tuple<int, int, int, int>> resource_pairs, 
                        std::unordered_map<int, Node>& nodes) {
    nodes.clear();

	//setup a vector to store the masks of the tasks
	//which are giving resources to other tasks
	std::vector<__uint128_t> task_masks;
    
    //create all nodes
    for (size_t i = 0; i < resource_pairs.size(); i++) {

		auto [x, y, z, w] = resource_pairs[i];
		nodes[i] = Node{(int)i, x, y, z, w, {}};

		task_masks.push_back(0);
    
	}

	//if system has barrier, just do it lazily
	if (barrier){

		for (int consumer_id = 0; consumer_id < (int) nodes.size(); consumer_id++){

			Node& consumer = nodes[consumer_id];
			int needed_a = -consumer.a;
			int needed_b = -consumer.b;
			int needed_c = -consumer.c;
			int needed_d = -consumer.d;
			
			for (int provider_id = 0; provider_id < (int) nodes.size(); provider_id++) {

				if (provider_id == consumer_id) continue;
				
				Node& provider = nodes[provider_id];
				Edge new_edge{consumer_id, 0, 0, 0, 0};
				
				//Try to satisfy a resource need
				if (needed_a > 0 && provider.a > 0) {

					int transfer = std::min(needed_a, provider.a);
					new_edge.a_amount = transfer;
					needed_a -= transfer;
					provider.a -= new_edge.a_amount;

					//place message in queue for the task giving up processors
					if (provider_id != (int) nodes.size() - 1)
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 0, transfer);

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}
				
				//Try to satisfy b resource need
				if (needed_b > 0 && provider.b > 0) {

					int transfer = std::min(needed_b, provider.b);
					new_edge.b_amount = transfer;
					needed_b -= transfer;
					provider.b -= new_edge.b_amount;
					

					//place message in queue for the task giving up processors
					if (provider_id != (int) nodes.size() - 1)
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 1, transfer);

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}
				
				//Try to satisfy c resource need
				if (needed_c > 0 && provider.c > 0) {

					int transfer = std::min(needed_c, provider.c);
					new_edge.c_amount = transfer;
					needed_c -= transfer;
					provider.c -= new_edge.c_amount;

					//place message in queue for the task giving up processors
					if (provider_id != (int) nodes.size() - 1)
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 2, transfer);

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}
				
				//Try to satisfy d resource need
				if (needed_d > 0 && provider.d > 0) {

					int transfer = std::min(needed_d, provider.d);
					new_edge.d_amount = transfer;
					needed_d -= transfer;
					provider.d -= new_edge.d_amount;
					

					//place message in queue for the task giving up processors
					if (provider_id != (int) nodes.size() - 1)
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 3, transfer);

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

			if (node.a == 0 && node.b == 0 && node.c == 0 && node.d == 0)
				discovered_consumers.push_back(i);

			//if a pure provider, add it to the list
			else if (node.a >= 0 && node.b >= 0 && node.c >= 0 && node.d >= 0)
				discovered_providers.push_back(i);

			//if it's a consumer, just add it to the discovered
			else if (node.a <= 0 && node.b <= 0 && node.c <= 0 && node.d <= 0)
				discovered_consumers.push_back(i);


			//if it's a transformer, add it to the transformer list
			else if ((node.a < 0 && node.b > 0) || (node.b < 0 && node.a > 0) || (node.c < 0 && node.d > 0) || (node.d < 0 && node.c > 0) || (node.a < 0 && node.c > 0) || (node.c < 0 && node.a > 0) || (node.a < 0 && node.d > 0) || (node.d < 0 && node.a > 0) || (node.b < 0 && node.c > 0) || (node.c < 0 && node.b > 0) || (node.b < 0 && node.d > 0) || (node.d < 0 && node.b > 0))
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
				int needed_a = -consumer.a;
				int needed_b = -consumer.b;
				int needed_c = -consumer.c;
				int needed_d = -consumer.d;

				int possible_a = 0;
				int possible_b = 0;
				int possible_c = 0;
				int possible_d = 0;

				//check if it's even possible to satisfy the transformer
				for (int& provider_id : discovered_providers) {

					Node& provider = nodes[provider_id];

					possible_a += provider.a;
					possible_b += provider.b;
					possible_c += provider.c;
					possible_d += provider.d;

				}

				//if not, then return
				if (needed_a > 0 && possible_a < needed_a)
					continue;

				else if (needed_b > 0 && possible_b < needed_b)
					continue;

				else if (needed_c > 0 && possible_c < needed_c)
					continue;

				else if (needed_d > 0 && possible_d < needed_d)
					continue;

				//otherwise satisfy the requirements
				for (int provider_id : discovered_providers) {
				
					Node& provider = nodes[provider_id];
					Edge new_edge{current_transformer, 0, 0, 0, 0};
					
					//Try to satisfy a resource need
					if (needed_a > 0 && provider.a > 0) {

						int transfer = std::min(needed_a, provider.a);
						new_edge.a_amount = transfer;
						needed_a -= transfer;
						provider.a -= new_edge.a_amount;

						//place message in queue for the task giving up processors
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(current_transformer, 0, transfer);

						//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << current_transformer << std::endl;

						//store the mask
						task_masks[current_transformer] |= ((__uint128_t) 1 << provider_id);

					}
					
					//Try to satisfy b resource need
					if (needed_b > 0 && provider.b > 0) {

						int transfer = std::min(needed_b, provider.b);
						new_edge.b_amount = transfer;
						needed_b -= transfer;
						provider.b -= new_edge.b_amount;

						//place message in queue for the task giving up processors
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(current_transformer, 1, transfer);

						//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << current_transformer << std::endl;

						//store the mask
						task_masks[current_transformer] |= ((__uint128_t) 1 << provider_id);

					}
					
					//Try to satisfy c resource need
					if (needed_c > 0 && provider.c > 0) {

						int transfer = std::min(needed_c, provider.c);
						new_edge.c_amount = transfer;
						needed_c -= transfer;
						provider.c -= new_edge.c_amount;

						//place message in queue for the task giving up processors
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(current_transformer, 2, transfer);

						//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << current_transformer << std::endl;

						//store the mask
						task_masks[current_transformer] |= ((__uint128_t) 1 << provider_id);

					}
					
					//Try to satisfy d resource need
					if (needed_d > 0 && provider.d > 0) {

						int transfer = std::min(needed_d, provider.d);
						new_edge.d_amount = transfer;
						needed_d -= transfer;
						provider.d -= new_edge.d_amount;

						//place message in queue for the task giving up processors
						schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(current_transformer, 3, transfer);

						//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << current_transformer << std::endl;

						//store the mask
						task_masks[current_transformer] |= ((__uint128_t) 1 << provider_id);

					}

				}

				//now this once transformer is a provider
				if (consumer.a < 0)
					consumer.a = 0;
				if (consumer.b < 0)
					consumer.b = 0;
				if (consumer.c < 0)
					consumer.c = 0;
				if (consumer.d < 0)
					consumer.d = 0;

				discovered_providers.push_back(current_transformer);

				current_transformer = -1;

				processed_transformers += 1;

			}

		}

		//now just do the same thing we did for transformers
		//but with the discovered consumers
		for (int consumer_id : discovered_consumers){

			Node& consumer = nodes[consumer_id];
			int needed_a = -consumer.a;
			int needed_b = -consumer.b;
			int needed_c = -consumer.c;
			int needed_d = -consumer.d;

			for (int provider_id : discovered_providers) {
			
				Node& provider = nodes[provider_id];
				Edge new_edge{consumer_id, 0, 0, 0, 0};
				
				//Try to satisfy a resource need
				if (needed_a > 0 && provider.a > 0) {

					int transfer = std::min(needed_a, provider.a);
					new_edge.a_amount = transfer;
					needed_a -= transfer;
					provider.a -= new_edge.a_amount;

					//place message in queue for the task giving up processors
					schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 0, transfer);

					//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << consumer_id << std::endl;

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}
				
				//Try to satisfy b resource need
				if (needed_b > 0 && provider.b > 0) {

					int transfer = std::min(needed_b, provider.b);
					new_edge.b_amount = transfer;
					needed_b -= transfer;
					provider.b -= new_edge.b_amount;

					//place message in queue for the task giving up processors
					schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 1, transfer);

					//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << consumer_id << std::endl;

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}
				
				//Try to satisfy c resource need
				if (needed_c > 0 && provider.c > 0) {

					int transfer = std::min(needed_c, provider.c);
					new_edge.c_amount = transfer;
					needed_c -= transfer;
					provider.c -= new_edge.c_amount;

					//place message in queue for the task giving up processors
					schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 2, transfer);

					//std::cerr << "Sending " << transfer << " processors from " << provider_id << " to " << consumer_id << std::endl;

					//store the mask
					task_masks[consumer_id] |= ((__uint128_t) 1 << provider_id);

				}
				
				//Try to satisfy d resource need
				if (needed_d > 0 && provider.d > 0) {

					int transfer = std::min(needed_d, provider.d);
					new_edge.d_amount = transfer;
					needed_d -= transfer;
					provider.d -= new_edge.d_amount;

					//place message in queue for the task giving up processors
					schedule.get_task(provider_id)->set_processors_to_send_to_other_processes(consumer_id, 3, transfer);

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

			if (provider.a > 0){

				//place message in queue for the task giving up processors
				schedule.get_task(discovered_providers[i])->set_processors_to_send_to_other_processes(nodes.size() - 1, 0, provider.a);

				//std::cerr << "Sending " << provider.a << " processors from " << discovered_providers[i] << " to free pool" << std::endl;

			}

			if (provider.b > 0){

				//place message in queue for the task giving up processors
				schedule.get_task(discovered_providers[i])->set_processors_to_send_to_other_processes(nodes.size() - 1, 1, provider.b);

				//std::cerr << "Sending " << provider.b << " processors from " << discovered_providers[i] << " to free pool" << std::endl;

			}
			
			if (provider.c > 0){

				//place message in queue for the task giving up processors
				schedule.get_task(discovered_providers[i])->set_processors_to_send_to_other_processes(nodes.size() - 1, 2, provider.c);

				//std::cerr << "Sending " << provider.c << " processors from " << discovered_providers[i] << " to free pool" << std::endl;

			}
			
			if (provider.d > 0){

				//place message in queue for the task giving up processors
				schedule.get_task(discovered_providers[i])->set_processors_to_send_to_other_processes(nodes.size() - 1, 3, provider.d);

				//std::cerr << "Sending " << provider.d << " processors from " << discovered_providers[i] << " to free pool" << std::endl;

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
			print_module::buffered_print(mode_strings, "Node ", id, " <", static_nodes[id].a, ",", static_nodes[id].b, ",", static_nodes[id].c, ",", static_nodes[id].d, "> → ");
		else
			print_module::buffered_print(mode_strings, "Free Resources", " <", static_nodes[id].a, ",", static_nodes[id].b, ",", static_nodes[id].c, ",", static_nodes[id].d, "> → ");

		if (node.edges.empty())
			print_module::buffered_print(mode_strings, "no edges");

		else {
			
			for (const Edge& edge : node.edges) {

				print_module::buffered_print(mode_strings, edge.to_node, "(");
				bool first = true;

				if (edge.a_amount > 0) {
				
				    print_module::buffered_print(mode_strings, "a:", edge.a_amount);
					first = false;
				
				}

				if (edge.b_amount > 0) {
				
				    if (!first) print_module::buffered_print(mode_strings, ",");
					print_module::buffered_print(mode_strings, "b:", edge.b_amount);
				
				}
				
				if (edge.c_amount > 0) {
				
				    if (!first) print_module::buffered_print(mode_strings, ",");
					print_module::buffered_print(mode_strings, "c:", edge.c_amount);
				
				}
				
				if (edge.d_amount > 0) {
				
				    if (!first) print_module::buffered_print(mode_strings, ",");
					print_module::buffered_print(mode_strings, "d:", edge.d_amount);
				
				}
				
				print_module::buffered_print(mode_strings, ") ");
			}
		}
		print_module::buffered_print(mode_strings, "\n");
	}
	print_module::buffered_print(mode_strings, "\n");

	print_module::flush(std::cerr, mode_strings);
}