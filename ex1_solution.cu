#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<cuda.h>

#define BLOCK_SIZE 256
#define ARRAY_SIZE 16777216

double getMillis(time_t t1, time_t t2) {
    return (double)(t2 - t1) * 1000.0L;
}

float sum(int n, float *f) {
    float res = 0.0f;
    for(int i = 0; i < n; i++) {
        res += f[i];
    }
    return res;
}

void cpu_saxpy(int n, float a, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

// Declare a kernel gpu_saxpy with the same args as cpu_saxpy
__global__ void gpu_saxpy(int n, float a, float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {

    float a = 2.0f;
    // Allocate arrays x and y with size ARRAY_SIZE
    float *x = (float*) malloc(sizeof(float) * ARRAY_SIZE);
    float *y = (float*) malloc(sizeof(float) * ARRAY_SIZE);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        x[i] = 0.1f;
        y[i] = 0.2f;
    }

    // Allocate arrays d_x and d_y and copy over content from host memory
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, sizeof(float) * ARRAY_SIZE);
    cudaMalloc((void**)&d_y, sizeof(float) * ARRAY_SIZE);
    cudaMemcpy(d_x, x, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);

    time_t t1 = clock();

    cpu_saxpy(ARRAY_SIZE, a, x, y);
    float error = sum(ARRAY_SIZE, y);
    
    time_t t2 = clock();
    
    // start your kernel here
    // make sure your execution configuration is correct
    int blocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_saxpy<<<blocks, BLOCK_SIZE>>>(ARRAY_SIZE, a, d_x, d_y);
    time_t t3 = clock();
    // copy the gpu memory back to the host
    cudaMemcpy(y, d_y, sizeof(float) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
    
    time_t t4 = clock();

    error = fabsf(error - sum(ARRAY_SIZE, y));
    if (error > 0.0001f) {
        printf("Error: The solution is not correct\n");
    } else {
        double cpuTime = getMillis(t1, t2);
        double gpuTime = getMillis(t3, t4);
        printf("Great job :)\n");
        printf("CPU Time: %.1fms\n", cpuTime);
        printf("GPU Time: %.1fms\n", gpuTime);
    }

    // release the memory of x and y
    free(x);
    free(y);
    // release the memory of d_x and d_y
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
