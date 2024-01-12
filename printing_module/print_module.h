#ifndef PRINTING_MODULE_H
#define PRINTING_MODULE_H

#include "print.h"
#include "print_buffer.h"

/*************************************************************************

print_module.h

This is the entrypoint for the print_module namespace and all related
functions. It provides a way to print to all standard C++ stream operators
as well as custom memory segment options which allows allocating, and printing
to a shared memory segment, or a set of segments.

In this file specifically, two creation methods for printable memory segments
are provided as well as two destruction methods

**************************************************************************/

namespace pm = print_module;
namespace print_module {

    print_buffer* createBuffer(std::string);

    std::vector<print_buffer*> createBuffer(buffer_set);

    int deleteBuffer(std::string);

    int deleteBuffer(buffer_set);

}

#endif