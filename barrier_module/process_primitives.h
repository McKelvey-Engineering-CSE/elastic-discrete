#ifndef PROCESS_PRIMITIVES_H
#define PROCESS_PRIMITIVES_H

#include <pthread.h>

class p_mutex {

private: 

    pthread_mutex_t mutex; 
    pthread_mutexattr_t mutexAttr;

    pthread_cond_t cond; 
    pthread_condattr_t condAttr;

public:

    p_mutex();
    ~p_mutex();
    void lock();
    void unlock();
    void wait();
    void notify_all();

};

#endif