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
