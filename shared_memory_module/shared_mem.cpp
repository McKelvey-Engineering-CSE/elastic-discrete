#include "shared_mem.h"
#include <unistd.h>
#include <sys/types.h>
#include <iostream>

#include "print_module.h"

//Guards are placed before and after the allocated shared memory region so we
//can periodically check for memory stomping, value is randomly generated
#define GUARD_VALUE  266684235

/********************************************************************************
INVALID_HANDLE,   //The requested file descriptor handle was invalid
CREATION_FAILURE, //Call to shm_open failed to return a file descriptor
RESIZE_FAILURE,   //Call to ftruncate failed to resize shared memory
MAP_FAILURE      //Call to mmap failed to return a valid pointer
********************************************************************************/

typedef enum {
	SUCCESS = 0,
	INVALID_HANDLE,   
	CREATION_FAILURE, 
	RESIZE_FAILURE,
	MAP_FAILURE
} error_code_t;

/********************************************************************************

This constructor performs the following operations:
	1) checks the std::string name as a valid file descriptor name
	2) constructs the access flags for shm_open
	3) constructs the mode flags for shm_open
	4) tries to call shm_open to exclusively create the file descriptor
	5) if shm_open fails (it will fail for all but one process) then we
		retry the call to shm_open without the creation flag
	6) if the call to shm_open succeeds then we call ftruncate to set the
		size of the shared memory region
	7) constructs mmap protection flags and behavior flags
	8) calls mmap to map the file into this process space
	9) if the constructing object is the owner of the file descriptor,
		then that object will call is_owner()
	10) all objects synchronize so that none will leave
		until the owner has finished calling is_owner()

Mapped memory layout:
guard1 -> guard (size = sizeof(unsigned))
ptr    -> user requested region (size = size_bytes)
guard2 -> guard
extras -> overhead struct (size = sizeof(overhead))
guard3 -> guard 

*********************************************************************************/

shared_mem::shared_mem(std::string proper, access_mode mode, size_t size_bytes) {

	//We add a little bit of padding for bookeeping purposes:
	//There is an overhead struct containing data values we want to use
	//We place three guards we can check periodically for data corruption
	size_t real_size = size_bytes + sizeof(unsigned)*3 + sizeof(overhead);

	//Construct the access permissions flag. We make two versions, so that
	//we can try shm_open with two different flag sets, because there is
	//a race condition as to who gets to create the actual file descriptor.
	int shm_open_create_access = (mode == READ_WRITE) ? O_RDWR | O_CREAT | O_EXCL : O_RDONLY | O_CREAT | O_EXCL;
	int shm_open_access = (mode == READ_WRITE) ? O_RDWR : O_RDONLY;
	int mmap_prot = (mode == READ_WRITE) ? PROT_READ | PROT_WRITE : PROT_READ;
	mode_t fd_access_mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP;

	//make sure we got a valid access mode
	if( mode != READ_WRITE && mode != READ_ONLY ){
		print_module::print(std::cout, "FATAL: Shared memory mode must be either READ_WRITE or READ_ONLY!\n");
		exit(INVALID_HANDLE);
	}

	//check that the handle has less than 254 characters and has exactly one 
	//forward slash at the start of the string. If there is not, append it there.
	if( (proper = (proper[0] == '/') ? proper : "/" + proper).length() > 254 ){
		print_module::print(std::cout, "FATAL: Shared memory handle must be less than 253 characters!\n");
		exit(INVALID_HANDLE);
	}
	
	//Check that there is only one forwards slash in the name and store if there is
	if(proper.find(std::string("/"), 1) != std::string::npos){
		print_module::print(std::cout, "FATAL: Shared memory handle must contain at most one forward slash, which if present must be the first character.\n");
		exit(INVALID_HANDLE);
	}
	else{
		this->name = proper;
	}

	//We can now try to exclusively create a file descriptor with shm_open
	int fd = shm_open( proper.c_str(), shm_open_create_access, fd_access_mode);

	if( fd == -1 ){
		this->owner = false;
		fd = shm_open( proper.c_str(), shm_open_access, fd_access_mode);

		if( fd == -1 ){
			print_module::print(std::cerr, "FATAL: shm_open failed! Reason: ", strerror(errno), "\n");
			exit(CREATION_FAILURE);
		}
	} 
	
	else {
		this->owner = true;

		if ( ftruncate(fd, (4096 * ((real_size/4096) + 1))) == -1 ){
			print_module::print(std::cerr, "FATAL: File descriptor could not be resized! Reason: ", strerror(errno), "\n");
			exit(RESIZE_FAILURE);
		}
	}

	//At this point all processes should have a file descriptor open, and
	//this file can be mapped into the local process memory space with mmap

	//mmap_flags specifies the behavior of mmap. MAP_SHARED allows access by
	//multiple processes, MAP_LOCKED locks pages in memory, and MAP_POPULATE
	//prefaults (caches) all pages in the mapping to avoid page faulting
	//later. The latter two flags are important for real-time performance.
	int mmap_flags = MAP_SHARED | MAP_LOCKED | MAP_POPULATE;	

	void* retptr = mmap(NULL, real_size, mmap_prot, mmap_flags, fd, 0);
	if( retptr == MAP_FAILED ){
		print_module::print(std::cerr, "FATAL: Call to mmap failed! Reason: ", strerror(errno), " \n");
		exit(MAP_FAILURE);
	}

	//We now have a shared memory region and a pointer! We map the memory
	//region as specified above. This is very ugly, because our map is
	//not composed of regularly sized items. The solution used below is to
	//compute everything off of a single-byte sized data type (char),
	//and use sizeof to ensure correctness
	this->guard1 = (unsigned*) retptr;
	this->ptr = (void*)((char*)retptr + sizeof(unsigned));
	this->guard2 = (unsigned*)((char*)retptr + sizeof(unsigned) + size_bytes);
	this->extras = (overhead*)((char*)retptr + sizeof(unsigned)*2 + size_bytes);
	this->guard3 = (unsigned*)((char*)retptr + sizeof(unsigned)*2 + size_bytes + sizeof(overhead));

	//Set up the guard regions:
	*(this->guard1) = GUARD_VALUE;
	*(this->guard2) = GUARD_VALUE;
	*(this->guard3) = GUARD_VALUE;

	//If we are the owner, then we perform any initialization 
	//as well as setting up the extras data structure. If we are not the 
	//owner, we block until the owner has finished setting up.
	extras->reference_count.fetch_add(1);

	std::unique_lock<std::mutex> lock(extras->init_mux);

	if( this->owner ){
		print_module::print(std::cerr, "Notifying all processes\n");
		extras->has_owner = true;
		extras->init_lock = 1;
		extras->cv.notify_all();
	}

	else{

		if (extras->init_lock == 0){
			print_module::print(std::cerr, "Waiting on notification\n");
			extras->cv.wait(lock, [this] {return !(extras->init_lock == 0);});
			print_module::print(std::cerr, "Freed from waiting on notification\n");
		}
	}
}

//Check the guard values against the supposed value, if any are modified,
//then something has probably gone wrong
bool shared_mem::valid(){
	return ( *guard1 == GUARD_VALUE &&
                 *guard2 == GUARD_VALUE &&
                 *guard3 == GUARD_VALUE);
}

void* shared_mem::getMapping(){
	return ptr;
}

bool shared_mem::is_owner(){
	return this->owner;
}

struct overhead* shared_mem::getOverhead(){
	return extras;
}

//Deconstructor must reference count and call shm_unlink
shared_mem::~shared_mem(){

	extras->reference_count.fetch_sub(1);

	if(extras->reference_count == 0) { 
	    shm_unlink(name.c_str());
	}

}

