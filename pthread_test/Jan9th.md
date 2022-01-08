1. 对于``task.CPU_ready``,我的理解是``pthread_scheduling()``线程会在调度线程时将其作为指令其他线程开始工作的信号。这样理解是否正确？举例: 有3个task A,B,C. ``pthread_scheduling()`` 使得``taskA.CPU_ready = true``, taskA 接收这个信号后随即执行运算。
2. 我没有理解``mutex``的概念。在调用``pthread_mutex_lock(&lock);``时哪些内存地址被禁止访问了？局部变量？全局变量？
3. 由之前两条引出的另外一个问题, 如果不同线程之间将``task.CPU_ready``作为线程间通信的渠道, 那是否会因为``mutex``造成没法访问这个信号？
4. 如果我们需要``pthread_scheduling()``这个线程来调度线程 那么系统自带的scheduler ``SCHED_FIFO``有什么用呢？

1. sudo ./main para-1.txt para-2.txt para-3.txt para-4.txt para-5.txt``


```C++

struct task
{
    bool done;
    bool ready;
    pthread_t thread;
    double create_time;//the absolute time that a task is created
}

unsigned N_task = 5;

struct task[N_task];

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;


thread_func()
{
    //set scheduling parameter
    struct sched_param param;
    param.sched_priority = 99;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    //set affinity to a CPU
    cpu_set_t cpuset1;
    CPU_ZERO(&cpuset1);
    CPU_SET(1, &cpuset1);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset1);
    //lock the mutex
    pthread_mutex_lock(&lock);
    //wait for the ready condition to come
    while (task.ready == false)
    {
        pthread_cond_wait(&cond, &lock);
    }
    //execute the computation
    usleep(1000);
    task.done = true;
    task.ready = false;
    //unlock the mutex
    pthread_mutex_unlock(&lock);
}

pthread_scheduling()
{
    //set affinity to a CPU different from the previous one
    cpu_set_t cpuset2;
    CPU_ZERO(&cpuset2);
    CPU_SET(2, &cpuset2);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset2);

    while(true)
    {
        //check all available tasks to be scheduled, find the thread with the earliest create time
        double earliest = INT_MAX;
        for(unsigned i = 0; i < N_task; ++i)
        {
            if(!task[i].done)
            {
                earliest = min(earliest,task[i].create_time);
            }
        }
        //set the thread ready condition to be true, wait for the thread function to complete
        task[i].ready = true;
        while(task[i].ready = true);
        //proceed to the next cycle
    }
}

int main()
{
    pthread_t tidp[N_task];
}
```