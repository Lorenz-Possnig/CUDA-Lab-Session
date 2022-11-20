#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<iostream>
#include<fstream>

#define TILE_WIDTH 32

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
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > HEIGHT || col > WIDTH) {
        return;
    }

    int i = 0;

    double zr = 0.0;
    double zi = 0.0;

    while(i < MAX_ITERATIONS && zr * zr + zi * zi < 4.0) {
        double fz = zr * zr - zi * zi + cr[col];
        zi = 2.0 * zr * zi + ci[row];
        zr = fz;
        i++;
    }

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

    //Image image;
    //image.height = HEIGHT;
    //image.width = WIDTH;
    RGB *image = (RGB*) malloc(sizeof(RGB) * HEIGHT * WIDTH);

    double minR = -2.0;
    double maxR = 1.0;
    double minI = -1.5;
    double maxI = 1.5;

    double *cr = (double*) malloc(sizeof(double) * WIDTH);
    double *ci = (double*) malloc(sizeof(double) * HEIGHT);

    fillValues(cr, WIDTH, minR, maxR);
    fillValues(ci, HEIGHT, minI, maxI);

    RGB *d_rgb;
    double *d_cr;
    double *d_ci;
    
    cudaError_t status;
    status = cudaMalloc((void**)&d_rgb, sizeof(RGB) * HEIGHT * WIDTH);
    status = cudaMalloc((void**)&d_cr, sizeof(double) * WIDTH);
    status = cudaMalloc((void**)&d_ci, sizeof(double) * HEIGHT);

    if (status != cudaSuccess) {
        printf("Something went wrong allocating memory\n");
    }

    status = cudaMemcpy(d_cr, cr, sizeof(double) * HEIGHT, cudaMemcpyHostToDevice);
    status = cudaMemcpy(d_ci, ci, sizeof(double) * HEIGHT, cudaMemcpyHostToDevice);

    if (status != cudaSuccess) {
        printf("Something went wrong copying memory to device\n");
    }

    time_t t2 = clock();

    printf("Allocating memory took %.2fms\n", getMillis(t1, t2));

    int blocks_x = (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH;
    int blocks_y = (HEIGHT + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 grid(blocks_x, blocks_y, 1);
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
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
    cudaMemcpy(image, d_rgb, sizeof(RGB) * HEIGHT * WIDTH, cudaMemcpyDeviceToHost);

    cudaFree(d_rgb);
    cudaFree(d_cr);
    cudaFree(d_ci);

    time_t t4 = clock();
    
    printf("Drawing image took %.2fms\n", getMillis(t3, t4));

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

    return 0;
}
