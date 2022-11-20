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

int main() {

    float a = 2.0f;
    // Allocate arrays x and y with size ARRAY_SIZE

    for (int i = 0; i < ARRAY_SIZE; i++) {
        x[i] = 0.1f;
        y[i] = 0.2f;
    }

    // Allocate arrays d_x and d_y and copy over content from host memory

    time_t t1 = clock();

    cpu_saxpy(ARRAY_SIZE, a, x, y);
    float error = sum(ARRAY_SIZE, y);
    
    time_t t2 = clock();
    
    // start your kernel here


    time_t t3 = clock();

    // copy the gpu memory back to the host
    
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
    
    // release the memory of d_x and d_y


    return 0;
}