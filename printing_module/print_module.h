#ifndef PRINTING_MODULE_H
#define PRINTING_MODULE_H

#include "print.h"
#include "printBuffer.h"

//short hand name
namespace pm = print_module;

namespace print_module {

    printBuffer* createBuffer(std::string);

    std::vector<printBuffer*> createBuffer(bufferSet);

    int deleteBuffer(std::string);

    int deleteBuffer(bufferSet);

}

#endif