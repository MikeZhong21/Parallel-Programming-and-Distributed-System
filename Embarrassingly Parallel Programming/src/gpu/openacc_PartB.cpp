#include <iostream>
#include <cmath>
#include <chrono>
#include <openacc.h>
#include <stdio.h>
#include "utils.hpp"

const int FILTER_SIZE = 3;
double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);
    // Apply the filter to the image
    unsigned char *filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    unsigned char *rChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    unsigned char *gChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    unsigned char *bChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        rChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        gChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        bChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    unsigned char *buffer = new unsigned char[width * height * num_channels];
    for (int i = 0; i < width * height * num_channels; i++)
    {
        buffer[i] = input_jpeg.buffer[i];
    }
     
#pragma acc enter data copyin(filteredImage[0 : width * height * num_channels], \
                buffer[0 : width * height * num_channels], \
                rChannel[0 : width * height], \
                gChannel[0 : width * height], \
                bChannel[0 : width * height], \
                filter[0:3][0:3])

#pragma acc update device(filteredImage[0 : width * height * num_channels], \
                buffer[0 : width * height * num_channels], \
                rChannel[0 : width * height], \
                gChannel[0 : width * height], \
                bChannel[0 : width * height], \
                filter[0:3][0:3])

    auto start_time = std::chrono::high_resolution_clock::now();
#pragma acc parallel num_gangs(1024) vector_length(256)
#pragma acc loop gang
    for (int h = 1; h < height - 1; h++)
    {
#pragma acc loop vector
        for (int w = 1; w < width - 1; w++)
        {
            
            filteredImage[(h * width + w) * num_channels]
                = static_cast<unsigned char>(std::round(
                rChannel[((h - 1) * width + (w - 1))] * filter[0][0] 
                + rChannel[((h - 1) * width + (w - 1)) + 1] * filter[0][1] 
                + rChannel[((h - 1) * width + (w - 1)) + 2]  * filter[0][2]
                + rChannel[((h - 1) * width + (w - 1)) + width]  * filter[1][0] 
                + rChannel[((h - 1) * width + (w - 1)) + width + 1] * filter[1][1] 
                + rChannel[((h - 1) * width + (w - 1)) + width + 2] * filter[1][2]
                + rChannel[((h - 1) * width + (w - 1)) + width * 2] * filter[2][0] 
                + rChannel[((h - 1) * width + (w - 1)) + width * 2 + 1] * filter[2][1] 
                + rChannel[((h - 1) * width + (w - 1)) + width * 2 + 2] * filter[2][2]));

            filteredImage[(h * width + w) * num_channels + 1]
                = static_cast<unsigned char>(std::round( 
                gChannel[((h - 1) * width + (w - 1))] * filter[0][0] 
                + gChannel[((h - 1) * width + (w - 1)) + 1] * filter[0][1] 
                + gChannel[((h - 1) * width + (w - 1)) + 2]  * filter[0][2]
                + gChannel[((h - 1) * width + (w - 1)) + width]  * filter[1][0] 
                + gChannel[((h - 1) * width + (w - 1)) + width + 1] * filter[1][1] 
                + gChannel[((h - 1) * width + (w - 1)) + width + 2] * filter[1][2]
                + gChannel[((h - 1) * width + (w - 1)) + width * 2] * filter[2][0] 
                + gChannel[((h - 1) * width + (w - 1)) + width * 2 + 1] * filter[2][1] 
                + gChannel[((h - 1) * width + (w - 1)) + width * 2 + 2] * filter[2][2]));

            filteredImage[(h * width + w) * num_channels + 2]
                = static_cast<unsigned char>(std::round(
                bChannel[((h - 1) * width + (w - 1))] * filter[0][0] 
                + bChannel[((h - 1) * width + (w - 1)) + 1] * filter[0][1] 
                + bChannel[((h - 1) * width + (w - 1)) + 2]  * filter[0][2]
                + bChannel[((h - 1) * width + (w - 1)) + width]  * filter[1][0] 
                + bChannel[((h - 1) * width + (w - 1)) + width + 1] * filter[1][1] 
                + bChannel[((h - 1) * width + (w - 1)) + width + 2] * filter[1][2]
                + bChannel[((h - 1) * width + (w - 1)) + width * 2] * filter[2][0] 
                + bChannel[((h - 1) * width + (w - 1)) + width * 2 + 1] * filter[2][1] 
                + bChannel[((h - 1) * width + (w - 1)) + width * 2 + 2] * filter[2][2]));

        }
    }


 
#pragma acc update self(filteredImage[0 : width * height * num_channels])

#pragma acc exit data copyout(filteredImage[0 : width * height * num_channels])

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] buffer;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
