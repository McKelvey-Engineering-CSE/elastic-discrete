#ifndef MEMORY_ALLOCATOR_H
#define MEMORY_ALLOCATOR_H

namespace shared_memory_module{

    template <class T, typename... Args>
    T* allocate(std::string bufferName, Args&&... args){

        //check if we have a segment open \ open if needed
        int fd = shm_open(bufferName.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        if( fd == -1 ){
            std::perror("ERROR: opening allocated memory for printing failed");
            return nullptr;
        }

        int ret_val = ftruncate(fd, sizeof(T));
        if( ret_val == -1 ){
            std::perror("ERROR: opening allocated memory for printing call to ftruncate failed");
            return nullptr;
        }

        T* object = new ((mmap(NULL, sizeof(T), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, fd, 0))) T{(std::forward<Args>(args))...};

        if (object == MAP_FAILED){
            std::perror("ERROR: opening allocated memory for printing call to mmap failed");
            return nullptr;
        }
        
        ret_val = close(fd);
        if( ret_val == -1 ){
            std::perror("WARNING: opening allocated memory for printing call to close file descriptor failed\n");
        }

        return object;
    }
} 

namespace smm = shared_memory_module;

#endif