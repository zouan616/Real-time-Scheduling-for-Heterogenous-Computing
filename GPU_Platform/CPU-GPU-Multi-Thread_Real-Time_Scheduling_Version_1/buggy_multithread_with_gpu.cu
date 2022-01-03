#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <unistd.h>

__global__ void one() { printf("1\n"); }

void *call_one(void *a) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);
  printf("Calling 1\n");
  one<<<2, 1024>>>();
  cudaDeviceSynchronize();
  return NULL;
}

__global__ void two() { printf("2\n"); }

void *call_two(void *a) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);
  printf("Calling 2\n");
  two<<<2, 1024>>>();
  cudaDeviceSynchronize();
  return NULL;
}

__global__ void three() { printf("3\n"); }

void *call_three(void *a) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);
  printf("Calling 3\n");
  three<<<2, 1024>>>();
  cudaDeviceSynchronize();
  return NULL;
}

__global__ void four() { printf("4\n"); }

void *call_four(void *a) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);
  printf("Calling 4\n");
  four<<<2, 1024>>>();
  cudaDeviceSynchronize();
  return NULL;
}

int main() {
  pthread_attr_t attr;
  pthread_attr_init(&attr);

  struct sched_param param = {99};
  pthread_attr_setschedparam(&attr, &param);
  pthread_attr_setschedpolicy(&attr, SCHED_FIFO);

  pthread_t thr[4];

  pthread_create(&thr[0], &attr, call_one, &mask);
  pthread_create(&thr[1], &attr, call_two, &mask);
  pthread_create(&thr[2], &attr, call_three, &mask);
  pthread_create(&thr[3], &attr, call_four, &mask);

  for (int i = 0; i < 4; ++i)
    pthread_join(thr[i], NULL);

  return 0;
}
