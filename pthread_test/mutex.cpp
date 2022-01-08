#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>


pthread_mutex_t mutex;

void *thr_fun(void *arg)
{
    pthread_mutex_lock(&mutex);
    char *no = (char *)arg;
    for (unsigned i = 0; i < 5; ++i)
    {
        printf("%s thread, i:%d\n", no, i);
        sleep(1);
    }
    pthread_mutex_unlock(&mutex);
}

int main()
{
    pthread_t tid1, tid2;
    pthread_mutex_init(&mutex, NULL);

    char* str1 = "No1";
    char* str2 = "No2";

    pthread_create(&tid2, NULL, thr_fun, str2);
    pthread_create(&tid1, NULL, thr_fun, str1);

    pthread_join(tid2, NULL);
    // pthread_join(tid1, NULL);

    pthread_mutex_destroy(&mutex);
    return 0;
}