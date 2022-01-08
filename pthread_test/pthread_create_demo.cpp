#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <sys/time.h>
#include <fstream>
#include <algorithm>
#include <pthread.h>
#include <errno.h>
#include <vector>
#include <sched.h>

#define N_subtask 5

void *thread_function(void *arg)
{
    int* ret = 0;
    char *c = (char *)arg;
    printf("parameter %s \n", c);
    
    for (unsigned i = 0; i < 10; ++i)
    {
        printf("loop %d \n", i);
        if (i == 5)
        {
            pthread_exit(ret);
        }
    }
    return ret;
}

int main()
{

    pthread_t tid;
    char* para = "thread: ";
    pthread_create(&tid, NULL, thread_function, para);   

    void* status = new long;
    pthread_join(tid,&status);
    printf("return %d\n",(long)status);
    delete status;
    return 0;
}