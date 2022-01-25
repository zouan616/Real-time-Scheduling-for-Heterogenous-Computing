#include "kernels.cu"
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define N 1000000
#define P 15000

void *call_one(void *a) {
  struct sched_param param;
  param.sched_priority = 99;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  struct timeval tv_start;
  struct timeval tv_end;
  long duration;

  cudaStream_t strm;
  cudaStreamCreate(&strm);

  char *h = (char *)malloc(N);
  char *d;
  cudaMalloc((void **)&d, N);

  while (1) {
    gettimeofday(&tv_start, NULL);
    printf("start time: %ld\n", tv_start.tv_usec);

    for (int i = 0; i < N; ++i)
      h[i] = rand() % 128;

    cudaMemcpy(d, h, N, cudaMemcpyHostToDevice);
    one<<<2, 1024, 0, strm>>>(N, d);
    cudaStreamSynchronize(strm);

    gettimeofday(&tv_end, NULL);
    duration = tv_end.tv_sec * 1000000 + tv_end.tv_usec -
               (tv_start.tv_sec * 1000000 + tv_start.tv_usec);
    printf("duration: %ld\n", duration);
    if (duration < P)
      usleep(P - duration);
    else
      throw "err";
  }

  free(h);
  cudaFree(d);
  return NULL;
}

void *call_two(void *a) {

  struct sched_param param;
  param.sched_priority = 98;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = (char *)malloc(N);
  for (int i = 0; i < N; ++i)
    h[i] = rand() % 128;

  char *d;
  cudaMalloc((void **)&d, N);
  cudaMemcpy(d, h, N, cudaMemcpyHostToDevice);
  two<<<2, 1024>>>(N, d);
  free(h);

  usleep(1);
  return NULL;
}

void *call_three(void *a) {

  struct sched_param param;
  param.sched_priority = 97;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = (char *)malloc(N);
  for (int i = 0; i < N; ++i)
    h[i] = rand() % 128;

  char *d;
  cudaMalloc((void **)&d, N);
  cudaMemcpy(d, h, N, cudaMemcpyHostToDevice);
  three<<<2, 1024>>>(N, d);
  free(h);

  usleep(1);
  return NULL;
}

void *call_four(void *a) {

  struct sched_param param;
  param.sched_priority = 96;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = (char *)malloc(N);
  for (int i = 0; i < N; ++i)
    h[i] = rand() % 128;

  char *d;
  cudaMalloc((void **)&d, N);
  cudaMemcpy(d, h, N, cudaMemcpyHostToDevice);
  four<<<2, 1024>>>(N, d);
  free(h);

  usleep(1);
  return NULL;
}

void *call_five(void *a) {

  struct sched_param param;
  param.sched_priority = 95;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = (char *)malloc(N);
  for (int i = 0; i < N; ++i)
    h[i] = rand() % 128;

  char *d;
  cudaMalloc((void **)&d, N);
  cudaMemcpy(d, h, N, cudaMemcpyHostToDevice);
  five<<<2, 1024>>>(N, d);
  free(h);

  usleep(1);
  return NULL;
}

void *call_six(void *a) {

  struct sched_param param;
  param.sched_priority = 94;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = (char *)malloc(N);
  for (int i = 0; i < N; ++i)
    h[i] = rand() % 128;

  char *d;
  cudaMalloc((void **)&d, N);
  cudaMemcpy(d, h, N, cudaMemcpyHostToDevice);
  six<<<2, 1024>>>(N, d);
  free(h);

  usleep(1);
  return NULL;
}

void *call_seven(void *a) {

  struct sched_param param;
  param.sched_priority = 93;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = (char *)malloc(N);
  for (int i = 0; i < N; ++i)
    h[i] = rand() % 128;

  char *d;
  cudaMalloc((void **)&d, N);
  cudaMemcpy(d, h, N, cudaMemcpyHostToDevice);
  seven<<<2, 1024>>>(N, d);
  free(h);

  usleep(1);
  return NULL;
}

void *call_eight(void *a) {

  struct sched_param param;
  param.sched_priority = 92;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(1, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = (char *)malloc(N);
  for (int i = 0; i < N; ++i)
    h[i] = rand() % 128;

  char *d;
  cudaMalloc((void **)&d, N);
  cudaMemcpy(d, h, N, cudaMemcpyHostToDevice);
  eight<<<2, 1024>>>(N, d);
  free(h);

  usleep(1);
  return NULL;
}

int main() {

  pthread_t thr[8];

  pthread_create(&thr[0], NULL, call_one, NULL);
  pthread_create(&thr[1], NULL, call_two, NULL);
  pthread_create(&thr[2], NULL, call_three, NULL);
  pthread_create(&thr[3], NULL, call_four, NULL);
  pthread_create(&thr[4], NULL, call_five, NULL);
  pthread_create(&thr[5], NULL, call_six, NULL);
  pthread_create(&thr[6], NULL, call_seven, NULL);
  pthread_create(&thr[7], NULL, call_eight, NULL);

  for (int i = 0; i < 8; ++i)
    pthread_join(thr[i], NULL);

  return 0;
}
