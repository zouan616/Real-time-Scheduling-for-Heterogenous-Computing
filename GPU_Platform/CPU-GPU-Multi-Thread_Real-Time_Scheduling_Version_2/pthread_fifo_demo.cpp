#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <string>
pthread_mutex_t mutex;


void *thr_fun(void *arg)
{
    int* argPack = (int*)arg;
    //set scheduling parameter
    struct sched_param param;
    param.sched_priority = 99;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    //set affinity to a CPU
    cpu_set_t cpuset1;
    CPU_ZERO(&cpuset1);
    CPU_SET(1, &cpuset1);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset1);
    
    printf("This is thread %d\n",*argPack);
    return NULL;
}

void *thr_fun1(void *arg)
{
    int* argPack = (int*)arg;
    //set scheduling parameter
    struct sched_param param;
    param.sched_priority = 95;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    //set affinity to a CPU
    cpu_set_t cpuset1;
    CPU_ZERO(&cpuset1);
    CPU_SET(1, &cpuset1);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset1);
    
    printf("This is thread %d\n",*argPack);
    return NULL;
}

int main()
{
    pthread_t tid1, tid2,tid3,tid4,tid5,tid6,tid7;
    int para1 = 1;
    int para2 = 2;
    int para3 = 3;
    int para4 = 4;
    int para5 = 5;
    int para6 = 6;
    int para7 = 7;
    pthread_create(&tid1, NULL, thr_fun, &para1);
    pthread_create(&tid2, NULL, thr_fun, &para2);
    pthread_create(&tid3, NULL, thr_fun, &para3);
    pthread_create(&tid4, NULL, thr_fun, &para4);
    pthread_create(&tid5, NULL, thr_fun, &para5);
    pthread_create(&tid6, NULL, thr_fun, &para6);
    pthread_create(&tid7, NULL, thr_fun, &para7);
    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);
    pthread_join(tid3, NULL);
    pthread_join(tid4, NULL);
    pthread_join(tid5, NULL);
    pthread_join(tid6, NULL);
    pthread_join(tid7, NULL);
    return 0;
}