#include "print_buffer.h"
#include "memory_allocator.h"

namespace print_module {

	print_buffer* print_buffer::openBuffer(std::string bufferName){

		//call the memory allocator
		print_buffer *buffer = shared_memory_module::allocate<print_buffer>(bufferName);

		if (buffer == nullptr){
			exit(-1);
		}

		return buffer;
	}

	void print_buffer::printToBuffer(std::string input){

		//grab lock
		std::lock_guard<std::mutex> lkguard(lock);

		//write
		buffer[position] = input;
		position += 1;
	}

	std::string print_buffer::dumpBuffer(){

		std::string ret_str = "";

		for (std::string i : buffer) {
			ret_str += i;
		}

		return ret_str;
	}

	std::vector<std::string> buffer_set::fetch(){
		return list_of_buffers;
	}
}