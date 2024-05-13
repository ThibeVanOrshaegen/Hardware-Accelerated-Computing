#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../Include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Include/stb_image_write.h"

clock_t start, stop;
double cpu_time;

float kernel[3][3] = {
    {1, 0, -1},
    {1, 0, -1},
    {1, 0, -1}
};

void applyConvolution2D(const unsigned char* image, unsigned char* output, int width, int height, int channels, const float* kernel, int kernelWidth, int kernelHeight) {
    int edgeX = kernelWidth / 2;
    int edgeY = kernelHeight / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int ch = 0; ch < channels; ch++) {
                float sum = 0.0;

                for (int ky = -edgeY; ky <= edgeY; ky++) {
                    for (int kx = -edgeX; kx <= edgeX; kx++) {
                        int ix = x + kx;
                        int iy = y + ky;

                        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                            sum += kernel[(ky + edgeY) * kernelWidth + (kx + edgeX)] * image[(iy * width + ix) * channels + ch];
                        }
                    }
                }
                int val = (int)sum;
                output[(y * width + x) * channels + ch] = (unsigned char)(val > 255 ? 255 : (val < 0 ? 0 : val));
            }
        }
    }
}

void grayscale(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int gray = 0;
            for (int c = 0; c < channels; c++) {
                gray += input[(y * width + x) * channels + c];
            }
            gray /= channels;
            for (int c = 0; c < channels; c++) {
                output[(y * width + x) * channels + c] = (unsigned char)gray;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Path: %s <image_path> [noOut]\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 0); 
    if (img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }

    unsigned char* grayscaleImg = (unsigned char*)malloc(width * height * channels);
    grayscale(img, grayscaleImg, width, height, channels);

    unsigned char* outputImg = (unsigned char*)malloc(width * height * channels);

    start = clock();

    applyConvolution2D(grayscaleImg, outputImg, width, height, channels, (float*)kernel, 3, 3);

    stop = clock();
    cpu_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f\n", cpu_time);

    if (argc >= 3 && strcmp(argv[2], "noOut") != 0) {
        char OutputPath[100];
        snprintf(OutputPath, sizeof(OutputPath), "%s-output.png", argv[1]);
        stbi_write_png(OutputPath, width, height, channels, outputImg, width * channels);
    }

    stbi_image_free(img);
    free(grayscaleImg);
    free(outputImg);

    return 0;
}
