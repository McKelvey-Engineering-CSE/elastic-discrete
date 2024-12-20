#ifndef _RT_SHARED_MEM_H_
#define _RT_SHARED_MEM_H_

/*************************************************************************

shared_mem.h

This object encapsulates the concept of a memory region shared between
processes. It provides for construction, access, and destruction of such
an object.

Enum : access_mode 
        This object encapsulates the concept of a memory region shared between
        processes. It provides for construction, access, and destruction of such
        an object.

Struct : overhead

        Values that we want to keep in the shared memory region for our own purposes

Class : shared_mem

        This class encapsulates the creation of and access to a region of shared
        memory. Multiple processes can share this memory assuming they use the same
        name and specify the same size (in bytes). 

**************************************************************************/

#include <sys/mman.h> //shm_open and mmap
#include <sys/stat.h> //for mode constants
#include <fcntl.h>    //for O_* constants (see man shm_open)
#include <errno.h>    //for errno and error reporting
#include <stdio.h>    //for printf()
#include <stdlib.h>   //for exit()
#include <string.h>   //for strerror

#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>

enum access_mode {READ_ONLY, READ_WRITE};

struct overhead {
	int			num_tasks;
        int 		        init_lock;
	std::mutex 		init_mux;
        std::condition_variable cv;
        bool 			has_owner;
        std::atomic<int> 	reference_count;
	std::atomic<int>	utility; //count tasks
};

class shared_mem{

public:
        shared_mem(std::string name, access_mode mode_, size_t size_bytes);
        ~shared_mem();

        void* getMapping();
        struct overhead* getOverhead();
        bool valid();
        void setTermination();

protected:

        /*This function is called if the current object is determined
        to be the owner. Thus, it is only called once for any group
        of processes that access the same named memory region.*/
        bool is_owner(); 

        std::string name;
        unsigned* guard1;
        unsigned* guard2;
        unsigned* guard3; //used to check for data corruption
        struct overhead* extras; //Points to a struct used for object implementation
        void* ptr; //Logical pointer to area returned by mmap (start of shared memory) 
        
        /*Specifies if this process was the one that successfully 
        created the shared memory object with shm_open*/
        bool  owner; 

        //specifies is a fault has occurred in a scheduled task and the task system must end
        //bypasses all clean memory checks 
        bool terminate = false;
};



#endif
