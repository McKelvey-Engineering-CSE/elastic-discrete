#include "print_module.h"
#include "memory_allocator.h"

namespace print_module { 
    print_buffer* createBuffer(std::string bufferName){

        //call the memory allocator
		print_buffer *buffer = shared_memory_module::allocate<print_buffer>(bufferName);

		if (buffer == nullptr){
			print_module::print(std::cerr, "Cannot continue, a print buffer could not be allocated\n");
			exit(-1);
		}

        return buffer;
    }

    std::vector<print_buffer*> createBuffer(buffer_set bufferNames){

        //vector to hold the return pointers to these segments
        //FIXME: FIND BETTER SOLUTION FOR ERROR RETURNS
        std::vector<print_buffer*> return_buffers;
        
        //get each unique buffer name from the set
        for (std::string name : bufferNames.fetch()){

            //call the memory allocator
            print_buffer *buffer = shared_memory_module::allocate<print_buffer>(name);

            if (buffer == nullptr){
                print_module::print(std::cerr, "Cannot continue, a print buffer could not be allocated\n");
                exit(-1);
            }

            return_buffers.push_back(buffer);
        }

        return return_buffers;
    }

    int deleteBuffer(std::string bufferName){

        int ret_val = shm_unlink(bufferName.c_str());

        if (ret_val == -1 && errno != ENOENT){
            std::perror("WARNING: deleteBuffer call to shm_unlink failed\n");
            return ret_val;
        }

        return 0;
    }

    int deleteBuffer(buffer_set bufferNames){

        //get each unique buffer name from the set
        for (std::string name : bufferNames.fetch()){

            int ret_val = shm_unlink(name.c_str());

            if (ret_val == -1 && errno != ENOENT){
                std::perror("WARNING: deleteBuffer call to shm_unlink failed\n");
                return ret_val;
            }
        }

        return 0;
    }

}

