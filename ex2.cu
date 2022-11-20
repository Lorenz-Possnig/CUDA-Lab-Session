#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<iostream>
#include<fstream>

#define BLOCK_WIDTH 32

#define WIDTH 2000
#define HEIGHT 2000

#define MAX_RGB 256

#define MAX_ITERATIONS 500

using namespace std; // bad practice, you didn't see this

void fillValues(double* c, double endRange, double minVal, double maxVal) {
    for(int i = 0; i < endRange; i++) {
        c[i] = (i / endRange) * (maxVal - minVal) + minVal;
    }
}

double getMillis(time_t t1, time_t t2) {
    return (double)(t2 - t1) * 1000.0L;
}

// our struct representing a color
typedef struct {
    unsigned int red;
    unsigned int green;
    unsigned int blue;
} RGB;

__global__ void drawMandelbrot(RGB *rgb, double *cr, double *ci) {
    // to find out if a complex number c is part of the mandelbrot set:
    // 1. declare numbers zi and zr as 0
    // 2. calculate fz as zr^2 - zi^2 + cr (real part of c)
    // 3. set zi to 2 * zi * zr + ci (imaginary part of c)
    // 4. set zr to fz
    // repeat steps 2 to 4 as as long as zr^2 + zi^2 is smaller than 4
    // OR we go over a fixed amount of iterations (MAX_ITERATIONS)
    // count the number of iterations performed (in int i)
    
    // naming for the variables used for coloring
    int i = 0;
    int row =
    int col =

    // your code here

    // code below only influences coloring, feel free to play around with it
    int r,g,b;
    /* this will make the set appear only blue

    // comment this line for a green set instead
    i = (int)((i / MAX_ITERATIONS) * MAX_RGB * MAX_RGB * MAX_RGB);

    b = i / (MAX_RGB * MAX_RGB);
    int tmp = i - b * MAX_RGB * MAX_RGB;
    r = tmp / MAX_RGB;
    g = tmp - r * MAX_RGB;
    */

	int max = MAX_RGB * MAX_RGB * MAX_RGB;
	double t = (double)i / MAX_ITERATIONS;
	i = (int)(t * (double)max);
	b = i / (MAX_RGB * MAX_RGB);
	int nn = i - b * MAX_RGB * MAX_RGB;
	r = nn / MAX_RGB;
	g = nn - r * MAX_RGB;

    int idx = row * WIDTH + col;
    rgb[idx].red = r;
    rgb[idx].green = g;
    rgb[idx].blue = b;
}

int main(int argc, char** argv) {

    time_t t1 = clock();

    // Allocate an array of type RGB* the size of the image
    // dimensions are defined with HEIGHT and WIDTH
    RGB *image =

    double minR = -2.0;
    double maxR = 1.0;
    double minI = -1.5;
    double maxI = 1.5;

    // Allocate arrays cr and ci with lengths WIDTH and HEIGHT respectively
    double *cr = 
    double *ci = 

    fillValues(cr, WIDTH, minR, maxR);
    fillValues(ci, HEIGHT, minI, maxI);

    RGB *d_rgb;
    double *d_cr;
    double *d_ci;
    
    cudaError_t status;
    // Allocate memory on the GPU for d_rgb, d_cr and d_ci
    status = 
    status = 
    status = 

    if (status != cudaSuccess) {
        printf("Something went wrong allocating memory\n");
    }

    // copy the contents of cr and ci to the device
    status = 
    status = 

    if (status != cudaSuccess) {
        printf("Something went wrong copying memory to device\n");
    }

    time_t t2 = clock();

    printf("Allocating memory took %.2fms\n", getMillis(t1, t2));

    int blocks_x = (WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    int blocks_y = (HEIGHT + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    dim3 grid(blocks_x, blocks_y, 1);
    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    drawMandelbrot<<<grid, block>>>(d_rgb, d_cr, d_ci);
    time_t t3 = clock();
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("Kernel did not launch correctly\n");
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d\n", status);
    }

    // copy d_rgb into image
    status = 

    if (status != cudaSuccess) {
        printf("Something went wrong copying memory to host\n");
    }

    // free d_rgb, d_cr, d_ci
    status = cudaFree(d_rgb);
    status = cudaFree(d_cr);
    status = cudaFree(d_ci);

    if (status != cudaSuccess) {
        printf("Something went wrong freeing device memory\n");
    }

    time_t t4 = clock();
    
    printf("Drawing image took %.2fms\n", getMillis(t3, t4));

    // The fantastic world of C++
    // This writes the rgb values to a ppm file
    time_t t5 = clock();
    ofstream fout("output_image.ppm");
    fout << "P3" << endl;
    fout << WIDTH << " " << HEIGHT << endl;
    fout << MAX_RGB << endl;

    for (int h = 0; h < HEIGHT; h++) {
        for (int w = 0; w < WIDTH; w += 1) {
            int i = h * WIDTH + w;
            fout << image[i].red << " " << image[i].green << " " << image[i].blue << " ";
       }
        fout << endl;
    }
    fout.close();
    time_t t6 = clock();

    printf("Writing image to disk took %.2fms\n", getMillis(t5, t6));

    // free image, cr and ci

    return 0;
}