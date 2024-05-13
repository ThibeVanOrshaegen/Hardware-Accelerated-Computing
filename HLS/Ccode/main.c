#define STB_IMAGE_IMPLEMENTATION
#include "../Include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Include/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

clock_t start, stop;
double cpu_time;

void printPixelValues(unsigned char* image, int width, int height, int channels) {
    // Print pixel values for the first few pixels in the image
    printf("Input Image Pixel Values:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < channels; j++) {
            printf("%d ", image[i * width * channels + j]);
        }
        printf("\n");
    }
}

void printOutputPixelValues(unsigned char* outputImg, int width, int height, int channels) {
    // Print pixel values for the first few pixels in the output image
    printf("Output Image Pixel Values:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < channels; j++) {
            printf("%d ", outputImg[i * (width/2) * channels + j]);
        }
        printf("\n");
    }
}

void Max_Pooling(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    for(int i = 0; i < height; i += 2) {
        for(int j = 0; j < width; j += 2) {
            for(int c = 0; c < channels; c++) {
                unsigned char max_val = 0;
                // Iterate over the 2x2 pooling window
                for(int m = 0; m < 2; m++) {
                    for(int n = 0; n < 2; n++) {
                        // Calculate the pixel position within the window
                        int pixel_pos = ((i + m) * width + (j + n)) * channels + c;
                        // Get the pixel value
                        unsigned char pixel_val = image[pixel_pos];
                        // Update max_val if the current pixel value is greater
                        if(pixel_val > max_val) {
                            max_val = pixel_val;
                        }
                    }
                }
                // Assign the maximum value to the output pixel
                output[(i/2) * (width/2) * channels + (j/2) * channels + c] = max_val;
            }
        }
    }
}

void Min_Pooling(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    for(int i = 1; i < height; i+=2)
    {
        for(int j = 1; j < width; j+=2)
        {
            for (int c = 0; c < channels; c++)
            {
                unsigned char min_val = 0;
                //initializes min_val with the pixel value at the top-left corner of the current 2x2 window.
                min_val = image[(i-1) * width * channels + (j-1) * channels + c];

                if (image[(i-1) * width * channels + j * channels + c] < min_val)
                {
                    min_val = image[(i-1) * width * channels + j * channels + c];
                }
                if (image[i * width * channels + (j-1) * channels + c] < min_val)
                {
                    min_val = image[i * width * channels + (j-1) * channels + c];
                }
                if (image[i * width * channels + j * channels + c] < min_val)
                {
                    min_val = image[i * width * channels + j * channels + c];
                }

                output[(i/2) * (width/2) * channels + (j/2) * channels + c] = min_val;
            }
        }
    }
}

void Average_Pooling(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    for(int i = 1; i < height; i+=2)
    {
        for(int j = 1; j < width; j+=2)
        {
            for (int c = 0; c < channels; c++)
            {
                unsigned int sum = 0;
                sum += image[(i-1) * width * channels + (j-1) * channels + c];
                sum += image[(i-1) * width * channels + j * channels + c];
                sum += image[i * width * channels + (j-1) * channels + c];
                sum += image[i * width * channels + j * channels + c];

                unsigned char average_val = sum / 4;

                output[(i/2) * (width/2) * channels + (j/2) * channels + c] = average_val;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Arguments %s <image_path>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printPixelValues(img, width, height, channels);

    unsigned char* outputImg = (unsigned char*)malloc((width/2 )* (height/2) * channels);

    start =clock();

    Max_Pooling(img, outputImg, width, height, channels);

    printOutputPixelValues(outputImg, width, height, channels);

    stop =clock();
    cpu_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Time: %f\n", cpu_time);

    char OutputPath[100];
    snprintf(OutputPath, sizeof(OutputPath), "%s-nieuw.png", argv[1]);
    stbi_write_png(OutputPath, (width/2), (height/2), channels, outputImg, (width/2) * channels);

    stbi_image_free(img);
    free(outputImg);

    return 0;
}
