#ifndef MEMORY_ALLOCATOR_H
#define MEMORY_ALLOCATOR_H
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <math.h>
#include <iostream>
#include <vector>

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

    template <typename T>
    key_t nameToKey(T name){
        std::vector<unsigned long long> partials;
        partials.push_back(0);
        unsigned long long value = 0;
        int p = 17;
        long m = 1000000009;
        int current_pos = 0;

        if (pow(p, name.length()-1) * name[name.length()-1] > (pow(2, 64) - 1)){
            std::cout << "ERROR: NAME FOR SEGMENT TOO LONG: " << name << " EXITING\n";
            exit(-1);
        }

        for (size_t i = 0; i < name.length(); i++){

            //assume no overflow
            partials[current_pos] += int(name[i]) * pow(p, i);

            //check for overflow
            if (partials[current_pos] > value){
                value = partials[current_pos];
            }

            else{
                partials[current_pos] = value;
                partials.push_back(0);
                current_pos += 1;
                partials[current_pos] += int(name[i]) * pow(p, i);
                value = partials[current_pos];
            }
        }

        //calc the partial mods and add them
        for (size_t i = 0; i < partials.size(); i++){
            partials.at(i) = partials.at(i) % m;
        }

        value = 0;
        unsigned long long pastValue = 0;

        for (size_t i = 0; i < partials.size(); i++){

            value += partials.at(i);

            //make sure we are not overflowing even here
            if (value < pastValue){
                std::cout << "ERROR: NAME FOR SEGMENT TOO LONG: " << name << " EXITING\n";
                exit(-1);
            }
            else{
                pastValue = value;
            }
        }

        return (key_t)(value & 0xFFFFFFFF);
    }

    template <class T, typename... Args>
    T* allocate(std::string bufferName, Args&&... args){

        T* object;

        int shmid = shmget(nameToKey<std::string>(bufferName), sizeof(T), IPC_CREAT | IPC_EXCL | 0666);
        void* memory_block = (T*)shmat(shmid, NULL, 0);
        object = new (memory_block) T{(std::forward<Args>(args))...};
            

        if (object == nullptr){
            std::perror("ERROR: opening allocated memory for printing call to shmget failed");
            return nullptr;
        }
        
        return object;
    }

    template <class T>
    T* fetch(std::string bufferName){

        T* object;
            
        int shmid = shmget(nameToKey<std::string>(bufferName), sizeof(T), 0666);
        object = (T*)shmat(shmid, NULL, 0);

        if (object == nullptr){
            std::perror("ERROR: fetching allocated memory for printing call to shmget failed");
            return nullptr;
        }
        
        return object;
    }

    template <class T>
    void delete_memory(std::string bufferName){
        
        //remove barrier object
        shmctl(shmget(shared_memory_module::nameToKey<std::string>(bufferName), sizeof(T), 0666), IPC_RMID, NULL);
    }

    template <class T>
    int detatch(T* mem_seg){
        
        //disconenct from memory
        return shmdt(mem_seg);
    }
} 

namespace smm = shared_memory_module;

#endif