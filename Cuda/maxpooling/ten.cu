#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
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
    if (argc < 11) {
        printf("Usage: %s <image_path1> <image_path2> ... <image_path10>\n", argv[0]);
        return 1;
    }

    int width[10], height[10], channels[10];
    unsigned char* img[10];
    unsigned char* grayImg[10];
    unsigned char* outputImg[10];

    // Load all images
    for (int i = 0; i < 10; i++) {
        img[i] = stbi_load(argv[i + 1], &width[i], &height[i], &channels[i], 0);
        if (img[i] == NULL) {
            printf("Error in loading the image: %s\n", argv[i + 1]);
            exit(1);
        }

        // Convert the input image to grayscale
        grayImg[i] = (unsigned char*)malloc(width[i] * height[i]);

        for (int j = 0; j < width[i] * height[i]; j++) {
            // Grayscale conversion formula: Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
            grayImg[i][j] = (unsigned char)(0.2989 * img[i][3 * j] + 0.5870 * img[i][3 * j + 1] + 0.1140 * img[i][3 * j + 2]);
        }

        outputImg[i] = (unsigned char*)malloc(width[i] * height[i] * channels[i]);
    }

    // Define your convolution kernel
    float kernel[9] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    start = clock();

    for (int i = 0; i < 10; i++) {
        unsigned char* d_img;
        cudaMalloc(&d_img, width[i] * height[i]);
        cudaMemcpy(d_img, grayImg[i], width[i] * height[i], cudaMemcpyHostToDevice);

        unsigned char* d_outputImg;
        cudaMalloc(&d_outputImg, width[i] * height[i] * channels[i]);

        float* d_kernel;
        cudaMalloc(&d_kernel, 3 * 3 * sizeof(float));
        cudaMemcpy(d_kernel, kernel, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((width[i] + BLOCK_SIZE - 1) / BLOCK_SIZE, (height[i] + BLOCK_SIZE - 1) / BLOCK_SIZE);

        applyConvolution<<<gridSize, blockSize>>>(d_img, d_outputImg, width[i], height[i], channels[i], d_kernel);

        cudaDeviceSynchronize();
        cudaMemcpy(outputImg[i], d_outputImg, width[i] * height[i] * channels[i], cudaMemcpyDeviceToHost);

        cudaFree(d_img);
        cudaFree(d_outputImg);
        cudaFree(d_kernel);
    }

    stop = clock();
    cpu_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f\n", cpu_time);

    // Save output images
    for (int i = 0; i < 10; i++) {
        char OutputPath[100];
        snprintf(OutputPath, sizeof(OutputPath), "%s-output.png", argv[i + 1]);
        stbi_write_png(OutputPath, width[i], height[i], channels[i], outputImg[i], width[i] * channels[i]);

        stbi_image_free(img[i]);
        free(outputImg[i]);
        free(grayImg[i]);
    }

    return 0;
}
