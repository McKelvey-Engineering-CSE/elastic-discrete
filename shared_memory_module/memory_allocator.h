#ifndef MEMORY_ALLOCATOR_H
#define MEMORY_ALLOCATOR_H

/*************************************************************************

memory_allocator.h

A template for constructing any object we want in a unified way. All objects
are memory mapped and then constructed in place before returning a pointer
to the object. 

The function takes in a class type for construction and it takes a memory 
segment name and any number of arguments which will be passed to the
constructor of the object for construction.

Functions: allocate <template> 

**************************************************************************/

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