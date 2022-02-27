The structure of ``schdeuling_experiment.cu`` is:
```C++

int thread_func(/* ... */)
{
    /* set the thread affinity to 2 different CPUs */
    while(1)
    {
        /* start timing */
        for(int i = 0;i<TASK_NUM;++i)
        {
            cpu_task(/* ... */,cpuTaskLength);// this part should run "cpuTaskLength" millisecond
            gpu_task(/* ... */,gpuTaskLength);// this part should run "gpuTaskLength" millisecond
        }
        /* stop timing */
        /* check if time has run past the ddl of this task, if so abort */
    }
}


int main()
{
    pthread_data_t pthreadDataMain[8];
    /* generate 8 pthreads */
    /* create cpu and gpu tasks of random length for each thread */
    for(int i = 0;i < 8;++i)
    {
        /* create and join these pthreads */
    }
}

```
The program outputs the ``cpuTaskLen``, ``gpuTaskLen``,``ddl`` of each thread, and ``sleep time / ddl``, which tells how much time is left after each task.
```
sleep time / ddl = 469684/575000 = 0.816842
sleep time / ddl = 319296/575000 = 0.555297
sleep time / ddl = 358130/787000 = 0.455057
sleep time / ddl = 385163/787000 = 0.489407
sleep time / ddl = 656199/1162000 = 0.564715
sleep time / ddl = 537682/815000 = 0.659733
sleep time / ddl = 348933/787000 = 0.443371
sleep time / ddl = 333070/575000 = 0.579252
sleep time / ddl = 536712/815000 = 0.658542
sleep time / ddl = 384361/787000 = 0.488388
sleep time / ddl = -195386/575000 = -0.339802
Thread time exceeded, abort!
```

``cputask.cu`` and ``gputask.cu`` is to verify that each iteration of cpu and gpu task take 1 millisecond.

```
> sudo ./cputask
duration is 1011
duration is 1002
duration is 998
...
> sudo ./gputask
duration is 1015
duration is 997
duration is 994
...
```