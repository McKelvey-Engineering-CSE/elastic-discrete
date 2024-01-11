#ifndef PRINTING_H
#define PRINTING_H

#include <sstream>

#include "printBuffer.h"

/*************************************************************************

printing.h

Function that ensures both thread safe printing
courtesy of std::cout and std::cerr, but allows
for raceless printing as well without locking

penalties: 

std::cout/cerr - 50 microseconds
printing to buffer - 50 microseconds to create buffer and print
                   - 40 microseconds to print to extant buffer
printf - 30 microseconds

benefits: 

Allows for operator overloading
in the future for printing without encroaching
on how the printing is handled. Could be replaced
with C++ 20 idiomatic methods, but I want to keep
this all C++17 at the absolute worst


Function :  print(ostream, args...)
            print(const char[], args...)
            print(bufferSet, args...)

**************************************************************************/

template <typename Arg, typename... Args>
static void print(std::ostream& out, Arg&& arg, Args&&... args){   
    
    //ostringstream seems to have lowest possible 
    //overhead
    std::ostringstream ss;
    
    //expander boilerplate
    ss << std::forward<Arg>(arg);
    using expander = int[];
    (void)expander{0, (void(ss << std::forward<Args>(args)), 0)...};
    
    //print to whatever we were given
    out << ss.str();
}

template <typename Arg, typename... Args>
static void print(const char bufferChar[], Arg&& arg, Args&&... args){

    //char to string
    std::string bufferName(bufferChar);

    //creates the buffer if it does not exist
    printBuffer* buffer = printBuffer::createBuffer(bufferName);
    if (buffer == nullptr)
        exit(-1);

    //FIXME: try to avoid stringstream while mimicing the function
    std::ostringstream ss;
    
    //expander boilerplate
    ss << std::forward<Arg>(arg);
    using expander = int[];
    (void)expander{0, (void(ss << std::forward<Args>(args)), 0)...};
    
    //print to whatever we were given
    buffer->printToBuffer(ss.str());
}

template <typename Arg, typename... Args>
static void print(bufferSet bufferNames, Arg&& arg, Args&&... args){

    //loop over all the handles in the buffer
    for (std::string name : bufferNames.fetch()){

        //creates the buffer if it does not exist
        printBuffer* buffer = printBuffer::createBuffer(name);
        if (buffer == nullptr)
            exit(-1);

        //FIXME: try to avoid stringstream while mimicing the function
        std::ostringstream ss;
        
        //expander boilerplate
        ss << std::forward<Arg>(arg);
        using expander = int[];
        (void)expander{0, (void(ss << std::forward<Args>(args)), 0)...};
        
        //print to whatever we were given
        buffer->printToBuffer(ss.str());
    }
}

#endif