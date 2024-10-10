#ifndef print_buffer_H
#define print_buffer_H

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

/*************************************************************************

print_buffer.h

This file contains everything related to the print buffers. A single buffer
can be created in memory and printed to, or a buffer_set object can be created
and printed to. The buffer_set object is just a collection of unique print
buffers represented by name that can be passed to other functions or processes
to share target print buffers

Objects : buffer_set
          print_buffer

**************************************************************************/

namespace print_module {

    class buffer_set {

        private:
            std::vector<std::string> list_of_buffers;

        public:
            template <typename Arg, typename... Args>
            buffer_set(std::string firstName, Arg&& arg, Args&&... args){
                
                //expander boilerplate
                list_of_buffers.push_back(firstName);
                list_of_buffers.push_back(std::forward<Arg>(arg));
                using expander = int[];
                (void)expander{0, (void(list_of_buffers.push_back(std::forward<Args>(args))), 0)...};

            }

            std::vector<std::string> fetch();

            friend std::ostream& operator<<(std::ostream& os, buffer_set const & inputSet){

                std::string list_name = "";

                for(std::string name : inputSet.list_of_buffers)
                    list_name += name + " ";

                return os << list_name;
            }
    };

    class print_buffer {

        private:
            int position = 0;
            std::mutex lock;
            std::vector<std::string> buffer = std::vector<std::string>(255, std::string(" ", 255));

        public:
            static print_buffer* openBuffer(std::string);
            void printToBuffer(std::string);
            std::string dumpBuffer();
    };

}
#endif