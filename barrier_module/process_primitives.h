#ifndef PROCESS_PRIMITIVES_H
#define PROCESS_PRIMITIVES_H

#include <pthread.h>

class p_condition_variable;

class p_mutex {

friend class p_condition_variable;

private: 

    pthread_mutex_t mutex; 
    pthread_mutexattr_t mutexAttr;

public:

    p_mutex();
    ~p_mutex();
    void lock();
    void unlock();

};

class p_condition_variable {

private: 

    pthread_cond_t cond; 
    pthread_condattr_t condAttr;

public:

    p_condition_variable();
    ~p_condition_variable();
    void wait(p_mutex mut, bool (*FunctionPtr)());
    void wait(p_mutex mut);
    void notify_all();

};

#endif