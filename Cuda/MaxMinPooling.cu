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

// CUDA kernel for average pooling
__global__ void Average_Pooling_CUDA(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < height/2 && j < width/2) {
        for (int c = 0; c < channels; c++) {
            unsigned int sum = 0;
            sum += image[2*i * width * channels + 2*j * channels + c];
            sum += image[2*i * width * channels + (2*j+1) * channels + c];
            sum += image[(2*i+1) * width * channels + 2*j * channels + c];
            sum += image[(2*i+1) * width * channels + (2*j+1) * channels + c];

            unsigned char average_val = sum / 4;

            output[i * (width/2) * channels + j * channels + c] = average_val;
        }
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

    unsigned char* outputImg = (unsigned char*)malloc((width/2) * (height/2) * channels);

    unsigned char *d_img, *d_outputImg;
    cudaMalloc((void**)&d_img, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImg, (width/2) * (height/2) * channels * sizeof(unsigned char));

    cudaMemcpy(d_img, img, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    clock_t start, stop;
    double cpu_time;

    start = clock();
    Min_Pooling_CUDA<<<numBlocks, threadsPerBlock>>>(d_img, d_outputImg, width, height, channels);
    cudaDeviceSynchronize();
    stop = clock();

    cpu_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f\n", cpu_time);

    cudaMemcpy(outputImg, d_outputImg, (width/2) * (height/2) * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    char OutputPath[100];
    snprintf(OutputPath, sizeof(OutputPath), "%s-output.png", argv[1]);
    stbi_write_png(OutputPath, (width/2), (height/2), channels, outputImg, (width/2) * channels);

    stbi_image_free(img);
    free(outputImg);
    cudaFree(d_img);
    cudaFree(d_outputImg);

    return 0;
}
