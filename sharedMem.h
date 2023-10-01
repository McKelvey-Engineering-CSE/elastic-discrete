#ifndef _RT_SHARED_MEM_H_
#define _RT_SHARED_MEM_H_

//This object encapsulates the concept of a memory region shared between
//processes. It provides for construction, access, and destruction of such
//an object.

#include <sys/mman.h> //shm_open and mmap
#include <sys/stat.h> //for mode constants
#include <fcntl.h>    //for O_* constants (see man shm_open)
#include <errno.h>    //for errno and error reporting
#include <stdio.h>    //for printf()
#include <stdlib.h>   //for exit()
#include <string.h>   //for strerror

#include <string>
#include <atomic>
enum access_mode {READ_ONLY, READ_WRITE};

//Values that we want to keep in the shared memory region for our own purposes
struct overhead {
	int			num_tasks;
	int 			init_lock;
        bool 			has_owner;
	unsigned int 		reference_count;
	std::atomic<int>	utility; //count tasks
 //	std::atomic<bool>	need_change; //Whether things have changed

  //	bool 		read;
  //	bool		written;
};

//This class encapsulates the creation of and access to a region of shared
//memory. Multiple processes can share this memory assuming they use the same
//name and specify the same size (in bytes). 
class sharedMem{
public:
//sharedMem();
sharedMem(std::string name, access_mode mode_, size_t size_bytes);
//void init(std::string name, access_mode mode_, size_t size_bytes);
~sharedMem();

//bool validate(); //Checks for memory underrun/overrun outside of or into the
                 //mapped region. If this function returns false it is a good
                 //indication that a memory management error has occurred.

void* getMapping();
//void* getMapping2();

struct overhead* getOverhead();

bool valid();

protected:

bool is_owner(); //This function is called if the current object is determined
                 //to be the owner. Thus, it is only called once for any group
                 //of processes that access the same named memory region.

std::string name;
unsigned* guard1;
unsigned* guard2;
unsigned* guard3; //used to check for data corruption

//Added for RCU-type behavior
//unsigned* guard4;

struct overhead* extras; //Points to a struct used for object implementation
void* ptr; //Logical pointer to area returned by mmap (start of shared memory) 


//Added for RCU-type lock-free behavior
//void* ptr2; //

bool  owner; //Specifies if this process was the one that successfully
                     //created the shared memory object with shm_open

//void swap_pointers();
};



#endif
