#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

pthread_mutex_t mutex;



struct para
{
    unsigned threadID;
    unsigned start;
    unsigned dur;
    unsigned ddl;
};


void *thr_fun(void *arg)
{
    pthread_mutex_lock(&mutex);
    struct para *argPack = (para *)arg;
    printf("threadID: %d\n",argPack->threadID);
    usleep(argPack->dur);
    pthread_mutex_unlock(&mutex);
}

int main()
{
    pthread_t tid1, tid2;
    pthread_mutex_init(&mutex, NULL);
    
    char* str1 = "No1";
    char* str2 = "No2";

    struct timeval tv;
    gettimeofday(&tv, NULL);
    time_ms_start_global = tv.tv_sec * 1000000 + tv.tv_usec;
    pthread_create(&tid1, NULL, thr_fun, str1);
    pthread_create(&tid2, NULL, thr_fun, str2);

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    pthread_mutex_destroy(&mutex);
    return 0;
}