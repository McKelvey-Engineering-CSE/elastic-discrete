#ifndef PRINTING_H
#define PRINTING_H

#include <sstream>

#include "print_buffer.h"

#if __cplusplus > 201703L

    #include <syncstream>

#endif

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


Function :  print_module::print(ostream, args...)
            print_module::print(const char[], args...)
            print_module::print(buffer_set, args...)

**************************************************************************/

namespace print_module {

    template <typename Arg, typename... Args>
    static void buffered_print(std::ostringstream& oss, Arg&& arg, Args&&... args){   
        
        //expander boilerplate
        oss << std::forward<Arg>(arg);
        using expander = int[];
        (void)expander{0, (void(oss << std::forward<Args>(args)), 0)...};
        
    }

    #if __cplusplus > 201703L

        template <typename Arg, typename... Args>
        static void print(std::ostream& out, Arg&& arg, Args&&... args){   
            
            //basic_osyncstream if we support it
            std::basic_osyncstream oss(out);
            
            //expander boilerplate
            oss << std::forward<Arg>(arg);
            using expander = int[];
            (void)expander{0, (void(oss << std::forward<Args>(args)), 0)...};
            
        }

        template <typename Arg, typename... Args>
        static void task_print(std::ostream& out, Arg&& arg, Args&&... args){   
            
            //basic_osyncstream if we support it
            std::basic_osyncstream oss(out);

            //print header
            oss << "(" << getpid() << ") ";
            
            //expander boilerplate
            oss << std::forward<Arg>(arg);
            using expander = int[];
            (void)expander{0, (void(oss << std::forward<Args>(args)), 0)...};
            
        }

        template <typename... Args>
        static void flush(std::ostream& out, std::ostringstream& buff, Args&&... args){
            
            //basic_osyncstream if we support it
            std::basic_osyncstream oss(out);
            
            //flush buffer first
            oss << buff.str();
            buff.clear();

            //expander boilerplate
            using expander = int[];
            (void)expander{0, (void(oss << std::forward<Args>(args)), 0)...};
            
        }

        template <typename Arg, typename... Args>
        static void print(const char bufferChar[], Arg&& arg, Args&&... args){

            //char to string
            std::string bufferName(bufferChar);

            //creates the buffer if it does not exist
            print_buffer* buffer = print_buffer::openBuffer(bufferName);
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
        static void print(buffer_set bufferNames, Arg&& arg, Args&&... args){

            //loop over all the handles in the buffer
            for (std::string name : bufferNames.fetch()){

                //creates the buffer if it does not exist
                print_buffer* buffer = print_buffer::openBuffer(name);
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

    #else

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
        static void task_print(std::ostream& out, Arg&& arg, Args&&... args){   
            
            //ostringstream seems to have lowest possible 
            //overhead
            std::ostringstream ss;

            //print header
            ss << "(" << getpid() << ") ";
            
            //expander boilerplate
            ss << std::forward<Arg>(arg);
            using expander = int[];
            (void)expander{0, (void(ss << std::forward<Args>(args)), 0)...};
            
            //print to whatever we were given
            out << ss.str();
        }

        template <typename... Args>
        static void flush(std::ostream& out, std::ostringstream& buff, Args&&... args){
            
            //basic_osyncstream if we support it
            std::ostringstream ss;
            
            //flush buffer first
            ss << buff.str();
            buff.clear();

            //expander boilerplate
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
            print_buffer* buffer = print_buffer::openBuffer(bufferName);
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
        static void print(buffer_set bufferNames, Arg&& arg, Args&&... args){

            //loop over all the handles in the buffer
            for (std::string name : bufferNames.fetch()){

                //creates the buffer if it does not exist
                print_buffer* buffer = print_buffer::openBuffer(name);
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

}
#endif