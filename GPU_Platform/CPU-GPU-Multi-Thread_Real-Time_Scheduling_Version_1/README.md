
---
### ***buggy_multithread_no_gpu.cu***
No gpu. Scheduling policy is set outside thread function (in `pthread_create()`). The cpu threads do not exhibit FIFO attribute.

### ***buggy_multithread_no_gpu.cu***
Now each cpu thread call a gpu kernel so that we can profile. From profiling result it seems that the cpu threads are in parallel, and that's why threads in ***buggy_multithread_no_gpu.cu*** do not exhibit FIFO attribute.
![](profile_1.png)

### TODO
Set scheduling policy in thread function.
Better not use cuda synchronize function, use `sleep()` or `usleep()` instead.

---
### ***multithread_sync_example.cu***
A multithread model with thread synchronization.

### Notes
- Generally:
    ```cpp
    pthread_mutex_lock(&mut);
    // while/if (condition == false)
    pthread_cond_wait(&cond, &mut);
    pthread_mutex_unlock(&mut);
    ...
    pthread_mutex_lock(&mut);
    // condition == true
    pthread_cond_signal/broadcast(&cond);
    pthread_mutex_unlock(&mut);
    ```

- `pthread_cond_wait(&cond, &mut)` mainly has 4 steps:
  1. enqueue thread to the waiting queue of cond
  2. unlock &mut
  3. wait to be waked
  4. lock &mut

  The key point is that both `wait` and `signal/broadcast` must be coupled with a pair of lock/unlock of the SAME mutex. For `wait`, the mutex ensures that the enqueue operation is atomic. For `signal/broadcast`, the operation must be ensured to take place AFTER the code path gets the mutex, that is, after `wait` completes enqueueing and releases the mutex, which means you can write such kind of code (although NOT recommended):
    ```cpp
    pthread_mutex_lock(&mut);
    // condition == true
    pthread_mutex_unlock(&mut);
    pthread_cond_signal/broadcast(&cond); // move it after unlock
    ```
  (Step iv. seems not necessary and therefore
    ```cpp
    pthread_mutex_lock(&mut);
    // while/if (condition == false)
    pthread_cond_wait(&cond, &mut);
    ```
  is enough. This step may be used to keep code style uniform.)

- In ***multithread_sync_example.cu***, all task threads' `pthread_cond_wait()` use the same parameter `&cond`. Therefore, in `scheduler()`, a completion of memcpy in whichever task will broadcast to all task threads waiting in `cond`'s queue, and wake all these threads. To filter out falsely waked task threads, use a while loop:
    ```cpp
    pthread_mutex_lock(&mut);
    while (task0[j].MEM_ready == false)
        pthread_cond_wait(&cond, &mut);
    pthread_mutex_unlock(&mut);
    ```
  so that these threads will enter the next while loop and wait again.

  However, waking threads will trigger context switch and reclaim cpu, which wastes cpu resources. Better declare different pthread_cond_t variables for each task.
