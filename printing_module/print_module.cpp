#include "print_module.h"

namespace print_module { 
    printBuffer* createBuffer(std::string bufferName){

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

    std::vector<printBuffer*> createBuffer(bufferSet bufferNames){

        //vector to hold the return pointers to these segments
        //FIXME: FIND BETTER SOLUTION FOR ERROR RETURNS
        std::vector<printBuffer*> return_buffers;
        
        //get each unique buffer name from the set
        for (std::string name : bufferNames.fetch()){

            int fd = shm_open(name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
            if( fd == -1 ){
                std::perror("ERROR: opening allocated memory for printing failed");
                return_buffers.push_back(nullptr);
            }

            int ret_val = ftruncate(fd, sizeof(printBuffer));
            if( ret_val == -1 ){
                std::perror("ERROR: opening allocated memory for printing call to ftruncate failed");
                return_buffers.push_back(nullptr);
            }

            printBuffer *buffer = new ((mmap(NULL, sizeof(printBuffer), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, fd, 0))) printBuffer();

            if (buffer == MAP_FAILED){
                std::perror("ERROR: opening allocated memory for printing call to mmap failed");
                return_buffers.push_back(nullptr);
            }
            
            ret_val = close(fd);
            if( ret_val == -1 ){
                std::perror("WARNING: opening allocated memory for printing call to close file descriptor failed\n");
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

    int deleteBuffer(bufferSet bufferNames){

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