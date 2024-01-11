#ifndef PRINTBUFFER_H
#define PRINTBUFFER_H

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <cstdlib>
#include <errno.h>
#include <time.h>

#include <iostream>
#include <cerrno>
#include <mutex>
#include <sstream>
#include <vector>

class printBuffer {

    private:
        int position = 0;
        std::mutex lock;
        std::vector<std::string> buffer = std::vector<std::string>(255, std::string(" ", 255));

    public:
        static printBuffer* createBuffer(std::string);
        static void deleteBuffer(std::string);
        void printToBuffer(std::string);
        std::string dumpBuffer();
};

#endif