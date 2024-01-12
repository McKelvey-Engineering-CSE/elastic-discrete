#include "printBuffer.h"
#include "memory_allocator.h"

namespace print_module {

	printBuffer* printBuffer::openBuffer(std::string bufferName){

		//call the memory allocator
		printBuffer *buffer = shared_memory_module::allocate<printBuffer>(bufferName);

		if (buffer == nullptr){
			exit(-1);
		}

		return buffer;
	}

	void printBuffer::printToBuffer(std::string input){

		//grab lock
		std::lock_guard<std::mutex> lkguard(lock);

		//write
		buffer[position] = input;
		position += 1;
	}

	std::string printBuffer::dumpBuffer(){

		std::string ret_str = "";

		for (std::string i : buffer) {
			ret_str += i;
		}

		return ret_str;
	}

	std::vector<std::string> bufferSet::fetch(){
		return list_of_buffers;
	}
}