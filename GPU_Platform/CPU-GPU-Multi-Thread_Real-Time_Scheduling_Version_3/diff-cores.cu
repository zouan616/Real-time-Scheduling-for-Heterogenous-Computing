#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define N 1000000
#define min(x, y) (x) < (y) ? (x) : (y)

__global__ void one(int n, char *a) {

  int stride = n / (gridDim.x * blockDim.x);
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int end = min(n - 1, start + stride - 1);

  for (int i = end - 2; i > start; --i)
    for (int j = start; j <= i; ++j)
      if (a[j + 1] * a[j + 1] + 4399 > a[j] * a[j] + 4399) {
        int tmp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = tmp;
      }

  if (a[start + 1] > a[start]) {
    int tmp = a[start];
    a[start] = a[start + 1];
    a[start + 1] = tmp;
  }
}

void *call_one(void *a) {

  struct sched_param param;
  param.sched_priority = 99;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(2, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = malloc(N);
  for (int i = 0; i < N; ++i)
    h[i] = rand() % 128;

  char *d;
  cudaMalloc((void **)&d, N);
  cudaMemcpy(d, h, N, cudaMemcpyHostToDevice);
  one<<<2, 1024>>>(N, d);
  free(h);

  usleep(1);
  return NULL;
}

__global__ void two(int n, char *a) {

  int stride = n / (gridDim.x * blockDim.x);
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int end = min(n - 1, start + stride - 1);

  for (int i = end - 2; i > start; --i)
    for (int j = start; j <= i; ++j)
      if (a[j + 1] * a[j + 1] + 4399 > a[j] * a[j] + 4399) {
        int tmp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = tmp;
      }

  if (a[start + 1] > a[start]) {
    int tmp = a[start];
    a[start] = a[start + 1];
    a[start + 1] = tmp;
  }
}

void *call_two(void *a) {

  struct sched_param param;
  param.sched_priority = 98;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(2, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = malloc(N);
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

__global__ void three(int n, char *a) {

  int stride = n / (gridDim.x * blockDim.x);
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int end = min(n - 1, start + stride - 1);

  for (int i = end - 2; i > start; --i)
    for (int j = start; j <= i; ++j)
      if (a[j + 1] * a[j + 1] + 4399 > a[j] * a[j] + 4399) {
        int tmp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = tmp;
      }

  if (a[start + 1] > a[start]) {
    int tmp = a[start];
    a[start] = a[start + 1];
    a[start + 1] = tmp;
  }
}

void *call_three(void *a) {

  struct sched_param param;
  param.sched_priority = 97;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(2, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = malloc(N);
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

__global__ void four(int n, char *a) {

  int stride = n / (gridDim.x * blockDim.x);
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int end = min(n - 1, start + stride - 1);

  for (int i = end - 2; i > start; --i)
    for (int j = start; j <= i; ++j)
      if (a[j + 1] * a[j + 1] + 4399 > a[j] * a[j] + 4399) {
        int tmp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = tmp;
      }

  if (a[start + 1] > a[start]) {
    int tmp = a[start];
    a[start] = a[start + 1];
    a[start + 1] = tmp;
  }
}

void *call_four(void *a) {

  struct sched_param param;
  param.sched_priority = 96;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(2, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = malloc(N);
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

__global__ void five(int n, char *a) {

  int stride = n / (gridDim.x * blockDim.x);
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int end = min(n - 1, start + stride - 1);

  for (int i = end - 2; i > start; --i)
    for (int j = start; j <= i; ++j)
      if (a[j + 1] * a[j + 1] + 4399 > a[j] * a[j] + 4399) {
        int tmp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = tmp;
      }

  if (a[start + 1] > a[start]) {
    int tmp = a[start];
    a[start] = a[start + 1];
    a[start + 1] = tmp;
  }
}

void *call_five(void *a) {

  struct sched_param param;
  param.sched_priority = 99;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(3, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = malloc(N);
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

__global__ void six(int n, char *a) {

  int stride = n / (gridDim.x * blockDim.x);
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int end = min(n - 1, start + stride - 1);

  for (int i = end - 2; i > start; --i)
    for (int j = start; j <= i; ++j)
      if (a[j + 1] * a[j + 1] + 4399 > a[j] * a[j] + 4399) {
        int tmp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = tmp;
      }

  if (a[start + 1] > a[start]) {
    int tmp = a[start];
    a[start] = a[start + 1];
    a[start + 1] = tmp;
  }
}

void *call_six(void *a) {

  struct sched_param param;
  param.sched_priority = 9;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(3, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = malloc(N);
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

__global__ void seven(int n, char *a) {

  int stride = n / (gridDim.x * blockDim.x);
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int end = min(n - 1, start + stride - 1);

  for (int i = end - 2; i > start; --i)
    for (int j = start; j <= i; ++j)
      if (a[j + 1] * a[j + 1] + 4399 > a[j] * a[j] + 4399) {
        int tmp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = tmp;
      }

  if (a[start + 1] > a[start]) {
    int tmp = a[start];
    a[start] = a[start + 1];
    a[start + 1] = tmp;
  }
}

void *call_seven(void *a) {

  struct sched_param param;
  param.sched_priority = 97;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(3, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = malloc(N);
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

__global__ void eight(int n, char *a) {

  int stride = n / (gridDim.x * blockDim.x);
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int end = min(n - 1, start + stride - 1);

  for (int i = end - 2; i > start; --i)
    for (int j = start; j <= i; ++j)
      if (a[j + 1] * a[j + 1] + 4399 > a[j] * a[j] + 4399) {
        int tmp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = tmp;
      }

  if (a[start + 1] > a[start]) {
    int tmp = a[start];
    a[start] = a[start + 1];
    a[start + 1] = tmp;
  }
}

void *call_eight(void *a) {

  struct sched_param param;
  param.sched_priority = 96;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(3, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);

  char *h = malloc(N);
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
