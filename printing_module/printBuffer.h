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

namespace print_module {

    class bufferSet {

        private:
            std::vector<std::string> list_of_buffers;

        public:
            template <typename Arg, typename... Args>
            bufferSet(std::string firstName, Arg&& arg, Args&&... args){
                
                //expander boilerplate
                list_of_buffers.push_back(firstName);
                list_of_buffers.push_back(std::forward<Arg>(arg));
                using expander = int[];
                (void)expander{0, (void(list_of_buffers.push_back(std::forward<Args>(args))), 0)...};

            }

            std::vector<std::string> fetch();

            friend std::ostream& operator<<(std::ostream& os, bufferSet const & inputSet){

                std::string list_name = "";

                for(std::string name : inputSet.list_of_buffers)
                    list_name += name + " ";

                return os << list_name;
            }
    };

    class printBuffer {

        private:
            int position = 0;
            std::mutex lock;
            std::vector<std::string> buffer = std::vector<std::string>(255, std::string(" ", 255));

        public:
            static printBuffer* openBuffer(std::string);
            void printToBuffer(std::string);
            std::string dumpBuffer();
    };

}
#endif