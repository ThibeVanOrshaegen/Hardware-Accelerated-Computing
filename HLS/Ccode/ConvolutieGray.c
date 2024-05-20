#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../Include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Include/stb_image_write.h"

clock_t beginTime, endTime;
double elapsedTime;

float convKernel[3][3] = {
    {1, 0, -1},
    {1, 0, -1},
    {1, 0, -1}
};

void performConvolution(const unsigned char* inputImg, unsigned char* resultImg, int imgWidth, int imgHeight, int imgChannels, const float* kernelMatrix, int kernelW, int kernelH) {
    int halfKernelW = kernelW / 2;
    int halfKernelH = kernelH / 2;

    for (int row = 0; row < imgHeight; row++) {
        for (int col = 0; col < imgWidth; col++) {
            for (int channel = 0; channel < imgChannels; channel++) {
                float accumulator = 0.0;

                for (int ky = -halfKernelH; ky <= halfKernelH; ky++) {
                    for (int kx = -halfKernelW; kx <= halfKernelW; kx++) {
                        int pixelX = col + kx;
                        int pixelY = row + ky;

                        if (pixelX >= 0 && pixelX < imgWidth && pixelY >= 0 && pixelY < imgHeight) {
                            accumulator += kernelMatrix[(ky + halfKernelH) * kernelW + (kx + halfKernelW)] * inputImg[(pixelY * imgWidth + pixelX) * imgChannels + channel];
                        }
                    }
                }
                int pixelValue = (int)accumulator;
                resultImg[(row * imgWidth + col) * imgChannels + channel] = (unsigned char)(pixelValue > 255 ? 255 : (pixelValue < 0 ? 0 : pixelValue));
            }
        }
    }
}

void convertToGrayscale(const unsigned char* sourceImg, unsigned char* grayImg, int imgWidth, int imgHeight, int imgChannels) {
    for (int y = 0; y < imgHeight; y++) {
        for (int x = 0; x < imgWidth; x++) {
            int grayValue = 0;
            for (int ch = 0; ch < imgChannels; ch++) {
                grayValue += sourceImg[(y * imgWidth + x) * imgChannels + ch];
            }
            grayValue /= imgChannels;
            for (int ch = 0; ch < imgChannels; ch++) {
                grayImg[(y * imgWidth + x) * imgChannels + ch] = (unsigned char)grayValue;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <image_path> [noOutput]\n", argv[0]);
        return 1;
    }

    int imgWidth, imgHeight, imgChannels;
    unsigned char* inputImage = stbi_load(argv[1], &imgWidth, &imgHeight, &imgChannels, 0);
    if (inputImage == NULL) {
        printf("Failed to load image\n");
        exit(1);
    }

    unsigned char* grayImage = (unsigned char*)malloc(imgWidth * imgHeight * imgChannels);
    convertToGrayscale(inputImage, grayImage, imgWidth, imgHeight, imgChannels);

    unsigned char* convolutedImage = (unsigned char*)malloc(imgWidth * imgHeight * imgChannels);

    beginTime = clock();

    performConvolution(grayImage, convolutedImage, imgWidth, imgHeight, imgChannels, (float*)convKernel, 3, 3);

    endTime = clock();
    elapsedTime = ((double)(endTime - beginTime)) / CLOCKS_PER_SEC;
    printf("Processing time: %f seconds\n", elapsedTime);

    char outputFilePath[100];
    snprintf(outputFilePath, sizeof(outputFilePath), "%s_convoluted.png", argv[1]);
    stbi_write_png(outputFilePath, imgWidth, imgHeight, imgChannels, convolutedImage, imgWidth * imgChannels);

    stbi_image_free(inputImage);
    free(grayImage);
    free(convolutedImage);

    return 0;
}
