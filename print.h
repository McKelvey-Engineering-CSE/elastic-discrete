#ifndef PRINTING_H
#define PRINTING_H

#include <sstream>

/*************************************************************************

printing.h

Function that ensures both thread safe printing
courtesy of std::cout and std::cerr, but allows
for raceless printing as well without locking

penalties: 

std::cout/cerr - 50 microseconds
printf - 30 microseconds

benefits: 

Allows for operator overloading
in the future for printing without encroaching
on how the printing is handled. Could be replaced
with C++ 20 idiomatic methods, but I want to keep
this all C++17 at the absolute worst


Function : cprint(args...)

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

#endif