#define STB_IMAGE_IMPLEMENTATION
#include "../Include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Include/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel for max pooling
__global__ void Max_Pooling_CUDA(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < height/2 && j < width/2) {
        for (int c = 0; c < channels; c++) {
            unsigned char max_val = 0;
            max_val = image[(2*i) * width * channels + (2*j) * channels + c];

            if (image[(2*i) * width * channels + (2*j+1) * channels + c] > max_val) {
                max_val = image[(2*i) * width * channels + (2*j+1) * channels + c];
            }
            if (image[(2*i+1) * width * channels + (2*j) * channels + c] > max_val) {
                max_val = image[(2*i+1) * width * channels + (2*j) * channels + c];
            }
            if (image[(2*i+1) * width * channels + (2*j+1) * channels + c] > max_val) {
                max_val = image[(2*i+1) * width * channels + (2*j+1) * channels + c];
            }

            output[i * (width/2) * channels + j * channels + c] = max_val;
        }
    }
}

// CUDA kernel for min pooling
__global__ void Min_Pooling_CUDA(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < height/2 && j < width/2) {
        for (int c = 0; c < channels; c++) {
            unsigned char min_val = 255; // Initialize with maximum possible value
            
            unsigned char val_00 = image[2*i * width * channels + 2*j * channels + c];
            unsigned char val_01 = image[2*i * width * channels + (2*j+1) * channels + c];
            unsigned char val_10 = image[(2*i+1) * width * channels + 2*j * channels + c];
            unsigned char val_11 = image[(2*i+1) * width * channels + (2*j+1) * channels + c];
            
            min_val = min(min(val_00, val_01), min(val_10, val_11));

            output[i * (width/2) * channels + j * channels + c] = min_val;
        }
    }
}
int main(int argc, char* argv[]) {
    if (argc < 11) {
        printf("Usage: %s <image1_path> <image2_path> ... <image10_path>\n", argv[0]);
        return 1;
    }

    const int num_images = 10;
    int width[num_images], height[num_images], channels[num_images];
    unsigned char* img[num_images];
    unsigned char* outputImg[num_images];

    unsigned char *d_img[num_images], *d_outputImg[num_images];

    dim3 threadsPerBlock(16, 16);

    clock_t start, stop;
    double cpu_time;

    for (int i = 0; i < num_images; i++) {
        img[i] = stbi_load(argv[i + 1], &width[i], &height[i], &channels[i], 0);
        if (img[i] == NULL) {
            printf("Error in loading image %d\n", i + 1);
            continue;
        }

        outputImg[i] = (unsigned char*)malloc((width[i] / 2) * (height[i] / 2) * channels[i]);

        cudaMalloc((void**)&d_img[i], width[i] * height[i] * channels[i] * sizeof(unsigned char));
        cudaMalloc((void**)&d_outputImg[i], (width[i] / 2) * (height[i] / 2) * channels[i] * sizeof(unsigned char));

        cudaMemcpy(d_img[i], img[i], width[i] * height[i] * channels[i] * sizeof(unsigned char), cudaMemcpyHostToDevice);

        dim3 numBlocks((width[i] + threadsPerBlock.x - 1) / threadsPerBlock.x, (height[i] + threadsPerBlock.y - 1) / threadsPerBlock.y);

        start = clock();
        Max_Pooling_CUDA<<<numBlocks, threadsPerBlock>>>(d_img[i], d_outputImg[i], width[i], height[i], channels[i]);
        cudaDeviceSynchronize();
        stop = clock();

        cpu_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
        printf("Time taken for image %d: %f\n", i + 1, cpu_time);

        cudaMemcpy(outputImg[i], d_outputImg[i], (width[i] / 2) * (height[i] / 2) * channels[i] * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        char OutputPath[100];
        snprintf(OutputPath, sizeof(OutputPath), "%s-output.png", argv[i + 1]);
        stbi_write_png(OutputPath, (width[i] / 2), (height[i] / 2), channels[i], outputImg[i], (width[i] / 2) * channels[i]);

        stbi_image_free(img[i]);
        free(outputImg[i]);
        cudaFree(d_img[i]);
        cudaFree(d_outputImg[i]);
    }

    return 0;
}
