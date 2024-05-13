#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Include/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../Include/stb_image.h"
#include <time.h>

clock_t start, stop;
double cpu_time;

#define BLOCK_SIZE 128

__global__ void applyConvolution(unsigned char* image, unsigned char* output, int width, int height, int channels, float* kernel) {
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    int x = threadid % width;
    int y = threadid / width;
    int edge = 1; // Since kernel size is 3x3

    while (y < height) {
        float sum[3] = {0.0, 0.0, 0.0}; // Sum for each channel
        /*Dit zou in een kernel moeten*/
        for (int ky = -edge; ky <= edge; ky++) {
            for (int kx = -edge; kx <= edge; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    for (int ch = 0; ch < channels; ch++) {
                        if (ch < 3) { // Apply convolution only to RGB channels
                            sum[ch] += kernel[(ky + edge) * 3 + (kx + edge)] * image[(iy * width + ix) * channels + ch];
                        }
                    }
                }
            }
        }
        /*tot hier*/
        for (int ch = 0; ch < channels; ch++) {
            if (ch < 3) {
                int val = (int)sum[ch];
                output[(y * width + x) * channels + ch] = (unsigned char)(val > 255 ? 255 : (val < 0 ? 0 : val));
            } else {
                // Preserve the alpha channel if present
                output[(y * width + x) * channels + ch] = 255;
            }
        }

        threadid += blockDim.x * gridDim.x;
        x = threadid % width;
        y = threadid / width;
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }

    // Convert the input image to grayscale
    unsigned char* grayImg = (unsigned char*)malloc(width * height);

    for (int i = 0; i < width * height; i++) {
        // Grayscale conversion formula: Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
        grayImg[i] = (unsigned char)(0.2989 * img[3 * i] + 0.5870 * img[3 * i + 1] + 0.1140 * img[3 * i + 2]);
    }

    // Define your convolution kernel
    float kernel[9] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    unsigned char* outputImg = (unsigned char*)malloc(width * height * channels);

    start =clock();

        unsigned char* d_img;
        cudaMalloc(&d_img, width * height);
        cudaMemcpy(d_img, grayImg, width * height, cudaMemcpyHostToDevice);

        unsigned char* d_outputImg;
        cudaMalloc(&d_outputImg, width * height * channels);

        float* d_kernel;
        cudaMalloc(&d_kernel, 3 * 3 * sizeof(float));
        cudaMemcpy(d_kernel, kernel, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
        applyConvolution<<<1023, 256>>>(d_img, d_outputImg, width, height, channels, d_kernel);
    
    
        cudaDeviceSynchronize();
        cudaMemcpy(outputImg, d_outputImg, width * height * channels, cudaMemcpyDeviceToHost);
    
    stop =clock();
    cpu_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f\n", cpu_time);

    if (strcmp(argv[2], "noOut") != 0) {
        char OutputPath[100];
        snprintf(OutputPath, sizeof(OutputPath), "%s-output.png", argv[1]);
        stbi_write_png(OutputPath, width, height, channels, outputImg, width * channels);
    }

    stbi_image_free(img);
    free(outputImg);
    free(grayImg);

    cudaFree(d_img);
    cudaFree(d_outputImg);
    cudaFree(d_kernel);

    return 0;
}
