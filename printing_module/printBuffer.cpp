#include "printBuffer.h"

namespace print_module {

	printBuffer* printBuffer::openBuffer(std::string bufferName){

		//check if we have a segment open \ open if needed
		int fd = shm_open(bufferName.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
		if( fd == -1 ){
			std::perror("ERROR: opening allocated memory for printing failed");
			return nullptr;
		}

		int ret_val = ftruncate(fd, sizeof(printBuffer));
		if( ret_val == -1 ){
			std::perror("ERROR: opening allocated memory for printing call to ftruncate failed");
			return nullptr;
		}

		printBuffer *buffer = new ((mmap(NULL, sizeof(printBuffer), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, fd, 0))) printBuffer();

		if (buffer == MAP_FAILED){
			std::perror("ERROR: opening allocated memory for printing call to mmap failed");
			return nullptr;
		}
		
		ret_val = close(fd);
		if( ret_val == -1 ){
			std::perror("WARNING: opening allocated memory for printing call to close file descriptor failed\n");
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