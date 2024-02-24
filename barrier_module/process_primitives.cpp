#include "process_primitives.h"


p_mutex::p_mutex(){

    //make process safe
    pthread_mutexattr_init(&mutexAttr);
    pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&mutex, &mutexAttr);
}

p_mutex::~p_mutex(){
    pthread_mutex_destroy(&mutex);
}

void p_mutex::lock(){
    pthread_mutex_lock(&mutex);
}

void p_mutex::unlock(){
    pthread_mutex_unlock(&mutex);
}


p_condition_variable::p_condition_variable(){

    //make process safe
    pthread_condattr_init(&condAttr);
    pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&cond, &condAttr);
}

p_condition_variable::~p_condition_variable(){
    pthread_cond_destroy(&cond);
}

void p_condition_variable::wait(p_mutex mut, bool (*FunctionPtr)()){

    while(FunctionPtr()){
        mut.lock();
        pthread_cond_wait(&cond, &mut.mutex);
        mut.unlock();
    }
}

void p_condition_variable::wait(p_mutex mut){

    mut.lock();
    pthread_cond_wait(&cond, &mut.mutex);
    mut.unlock();
}

void p_condition_variable::notify_all(){
    pthread_cond_broadcast(&cond);
}