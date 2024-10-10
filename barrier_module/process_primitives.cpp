#include "process_primitives.h"


p_mutex::p_mutex(){

    //make process safe
    pthread_mutexattr_init(&mutexAttr);
    pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&mutex, &mutexAttr);

    //make process safe
    pthread_condattr_init(&condAttr);
    pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&cond, &condAttr);
}

p_mutex::~p_mutex(){
    pthread_mutex_destroy(&mutex);
}

void p_mutex::lock(){
    pthread_mutex_lock(&mutex);
}

void p_mutex::wait(){

    pthread_cond_wait(&cond, &mutex);
    pthread_mutex_unlock(&mutex);

}

void p_mutex::notify_all(){
    pthread_cond_broadcast(&cond);
}

void p_mutex::unlock(){
    pthread_mutex_unlock(&mutex);
}