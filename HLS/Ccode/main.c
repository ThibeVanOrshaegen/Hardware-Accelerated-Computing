#define STB_IMAGE_IMPLEMENTATION
#include "../Include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Include/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

clock_t tStart, tEnd;
double elapsedTime;

void displayInputPixels(unsigned char* img, int imgWidth, int imgHeight, int colorChannels) {
    printf("Input Image Pixel Values:\n");
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < colorChannels; y++) {
            printf("%d ", img[x * imgWidth * colorChannels + y]);
        }
        printf("\n");
    }
}

void displayOutputPixels(unsigned char* outImg, int imgWidth, int imgHeight, int colorChannels) {
    printf("Output Image Pixel Values:\n");
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < colorChannels; y++) {
            printf("%d ", outImg[x * (imgWidth/2) * colorChannels + y]);
        }
        printf("\n");
    }
}

void maxPooling(unsigned char* img, unsigned char* outImg, int imgWidth, int imgHeight, int colorChannels) {
    for (int row = 0; row < imgHeight; row += 2) {
        for (int col = 0; col < imgWidth; col += 2) {
            for (int ch = 0; ch < colorChannels; ch++) {
                unsigned char maxVal = 0;
                for (int m = 0; m < 2; m++) {
                    for (int n = 0; n < 2; n++) {
                        int pixelIdx = ((row + m) * imgWidth + (col + n)) * colorChannels + ch;
                        unsigned char pixelVal = img[pixelIdx];
                        if (pixelVal > maxVal) {
                            maxVal = pixelVal;
                        }
                    }
                }
                outImg[(row/2) * (imgWidth/2) * colorChannels + (col/2) * colorChannels + ch] = maxVal;
            }
        }
    }
}

void minPooling(unsigned char* img, unsigned char* outImg, int imgWidth, int imgHeight, int colorChannels) {
    for (int row = 1; row < imgHeight; row += 2) {
        for (int col = 1; col < imgWidth; col += 2) {
            for (int ch = 0; ch < colorChannels; ch++) {
                unsigned char minVal = img[(row-1) * imgWidth * colorChannels + (col-1) * colorChannels + ch];
                if (img[(row-1) * imgWidth * colorChannels + col * colorChannels + ch] < minVal) {
                    minVal = img[(row-1) * imgWidth * colorChannels + col * colorChannels + ch];
                }
                if (img[row * imgWidth * colorChannels + (col-1) * colorChannels + ch] < minVal) {
                    minVal = img[row * imgWidth * colorChannels + (col-1) * colorChannels + ch];
                }
                if (img[row * imgWidth * colorChannels + col * colorChannels + ch] < minVal) {
                    minVal = img[row * imgWidth * colorChannels + col * colorChannels + ch];
                }
                outImg[(row/2) * (imgWidth/2) * colorChannels + (col/2) * colorChannels + ch] = minVal;
            }
        }
    }
}

void avgPooling(unsigned char* img, unsigned char* outImg, int imgWidth, int imgHeight, int colorChannels) {
    for (int row = 1; row < imgHeight; row += 2) {
        for (int col = 1; col < imgWidth; col += 2) {
            for (int ch = 0; ch < colorChannels; ch++) {
                unsigned int totalSum = 0;
                totalSum += img[(row-1) * imgWidth * colorChannels + (col-1) * colorChannels + ch];
                totalSum += img[(row-1) * imgWidth * colorChannels + col * colorChannels + ch];
                totalSum += img[row * imgWidth * colorChannels + (col-1) * colorChannels + ch];
                totalSum += img[row * imgWidth * colorChannels + col * colorChannels + ch];
                unsigned char avgVal = totalSum / 4;
                outImg[(row/2) * (imgWidth/2) * colorChannels + (col/2) * colorChannels + ch] = avgVal;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    int imgWidth, imgHeight, colorChannels;
    unsigned char* image = stbi_load(argv[1], &imgWidth, &imgHeight, &colorChannels, 0);
    if (image == NULL) {
        printf("Failed to load image\n");
        exit(1);
    }
    displayInputPixels(image, imgWidth, imgHeight, colorChannels);

    unsigned char* outImage = (unsigned char*)malloc((imgWidth/2) * (imgHeight/2) * colorChannels);

    tStart = clock();

    maxPooling(image, outImage, imgWidth, imgHeight, colorChannels);

    displayOutputPixels(outImage, imgWidth, imgHeight, colorChannels);

    tEnd = clock();
    elapsedTime = ((double)(tEnd - tStart)) / CLOCKS_PER_SEC;
    printf("Elapsed Time: %f seconds\n", elapsedTime);

    char outputPath[100];
    snprintf(outputPath, sizeof(outputPath), "%s-processed.png", argv[1]);
    stbi_write_png(outputPath, (imgWidth/2), (imgHeight/2), colorChannels, outImage, (imgWidth/2) * colorChannels);

    stbi_image_free(image);
    free(outImage);

    return 0;
}

