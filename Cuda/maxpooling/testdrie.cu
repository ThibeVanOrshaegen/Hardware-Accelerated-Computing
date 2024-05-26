#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h> 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Include/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../Include/stb_image.h"
#include <time.h>

clock_t start, stop;
double cpu_time;

#define BLOCK_SIZE 128

__global__ void applyConvolution(unsigned char* image, unsigned char* output, int width, int height, int channels, float* kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int edge = 1; // Since kernel size is 3x3
        float sum[3] = {0.0, 0.0, 0.0}; // Sum for each channel

        for (int ky = -edge; ky <= edge; ky++) {
            for (int kx = -edge; kx <= edge; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                
                // Check if the pixel is within the image bounds
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    for (int ch = 0; ch < channels; ch++) {
                        sum[ch] += kernel[(ky + edge) * 3 + (kx + edge)] * image[(iy * width + ix) * channels + ch];
                    }
                }
            }
        }
        
        // Write the convolution result to the output image
        for (int ch = 0; ch < channels; ch++) {
            int val = (int)sum[ch];
            output[(y * width + x) * channels + ch] = (unsigned char)(val > 255 ? 255 : (val < 0 ? 0 : val));
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <image_path1> <image_path2> ... <image_pathN>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    int imageCount = argc - 1;

    for (int i = 0; i < imageCount; ++i) {
        unsigned char* img = stbi_load(argv[i + 1], &width, &height, &channels, 0);
        if (img == NULL) {
            printf("Error in loading the image: %s\n", argv[i + 1]);
            continue;
        }

        float kernel[3][3] = {
            {1, 0, -1},
            {1, 0, -1},
            {1, 0, -1}
        };

        unsigned char* outputImg = (unsigned char*)malloc(width * height * channels);

        start = clock();

        unsigned char* d_img;
        cudaMalloc(&d_img, width * height * channels);
        cudaMemcpy(d_img, img, width * height * channels, cudaMemcpyHostToDevice);

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

        stop = clock();
        cpu_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
        printf("Time taken for image %d: %f\n", i + 1, cpu_time);

        char OutputPath[100];
        snprintf(OutputPath, sizeof(OutputPath), "%s-output.png", argv[i + 1]);
        stbi_write_png(OutputPath, width, height, channels, outputImg, width * channels);

        stbi_image_free(img);
        free(outputImg);

        cudaFree(d_img);
        cudaFree(d_outputImg);
        cudaFree(d_kernel);
    }

    return 0;
}
