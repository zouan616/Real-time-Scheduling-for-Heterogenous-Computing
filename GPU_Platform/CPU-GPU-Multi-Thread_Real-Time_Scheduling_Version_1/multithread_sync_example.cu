#include <pthread.h>
#include <sched.h>
...

// 3 tasks, each has 5 subtasks
#define N_task 3
#define N_subtask 5

    // which subtask that task[i] is executing
    int subtask_position[N_task];

// data computed in a subtask
struct Subtask {
  ...
};
Subtask task0[N_subtask], task1[N_subtask], task2[N_subtask];

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

/* ============================================= */

void *thread0(void *data) {

  // set scheduler ...
  // pin to a core ...

  for (int j = 0; j < N_subtask; j++) {

    // task 0 at j_th subtask
    subtask_position[0] = j;

    // GPU tasks ...
    // CPU tasks ...

    // CPU tasks completed, inform scheduler
    task0[j].CPU_ready = true;

    // wait scheduler copy data to cpu
    // before moving to next subtask
    // MEM_ready: whether copy completed
    pthread_mutex_lock(&lock);
    while (task0[j].MEM_ready == false)
      pthread_cond_wait(&cond, &lock);
    task0[j].MEM_ready = false;
    pthread_mutex_unlock(&lock);
  }

  // ...
  return 0;
}

void *thread1(void *data) {
  // similar
}

void *thread2(void *data) {
  // similar
}

void *scheduler(void *data) {

  // set scheduler ...
  // pin to a core DIFFERENT FROM task threads

  int i = 0;
  while (i < 500) {
    if (task0[subtask_position[0]].CPU_ready == true) {

      // task 0 completed its current subtask
      // cuda memcpy device to host async ...

      pthread_mutex_lock(&lock);
      task0[task_position[0]].MEM_ready = true;
      task0[task_position[0]].CPU_ready = false;
      pthread_cond_broadcast(&cond);
      pthread_mutex_unlock(&lock);
      i++;

    } else if (task1[subtask_position[1]].CPU_ready == true) {
      // similar
    } else if (task2[subtask_position[2]].CPU_ready == true) {
      // similar
    }
  }
}

int main() {

  // ...

  pthread_create(thread0);
  pthread_create(thread1);
  pthread_create(thread2);
  pthread_create(scheduler);

  // ...
  return 0;
}
