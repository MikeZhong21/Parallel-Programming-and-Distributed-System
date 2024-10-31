#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>    

#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
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
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int height = 1; height < input_jpeg.height - 1; height++)
    {
        for (int width = 1; width < input_jpeg.width - 1; width++)
        {
            int sum_r = 0, sum_g = 0, sum_b = 0;
            
            unsigned char *r_value = &input_jpeg.buffer[((height - 1) * input_jpeg.width + (width - 1)) * input_jpeg.num_channels];
            unsigned char *g_value = r_value + 1;
            unsigned char *b_value = r_value + 2;
            
            sum_r = *r_value * filter[0][0] + *(r_value + input_jpeg.num_channels) * filter[0][1] + *(r_value + input_jpeg.num_channels*2) * filter[0][2]
                + *(r_value + input_jpeg.width * input_jpeg.num_channels) * filter[1][0] + *(r_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels) * filter[1][1] + *(r_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[1][2]
                + *(r_value + input_jpeg.width * 2 * input_jpeg.num_channels) * filter[2][0] + *(r_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels) * filter[2][1] + *(r_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[2][2];
            sum_g = *g_value * filter[0][0] + *(g_value + input_jpeg.num_channels) * filter[0][1] + *(g_value + input_jpeg.num_channels*2) * filter[0][2]
                + *(g_value + input_jpeg.width * input_jpeg.num_channels) * filter[1][0] + *(g_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels) * filter[1][1] + *(g_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[1][2]
                + *(g_value + input_jpeg.width * 2 * input_jpeg.num_channels) * filter[2][0] + *(g_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels) * filter[2][1] + *(g_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[2][2];
            sum_b = *b_value * filter[0][0] + *(b_value + input_jpeg.num_channels) * filter[0][1] + *(b_value + input_jpeg.num_channels*2) * filter[0][2]
                + *(b_value + input_jpeg.width * input_jpeg.num_channels) * filter[1][0] + *(b_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels) * filter[1][1] + *(b_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[1][2]
                + *(b_value + input_jpeg.width * 2 * input_jpeg.num_channels) * filter[2][0] + *(b_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels) * filter[2][1] + *(b_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[2][2];
        
            filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels]
                = static_cast<unsigned char>(std::round(sum_r));
            filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels + 1]
                = static_cast<unsigned char>(std::round(sum_g));
            filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels + 2]
                = static_cast<unsigned char>(std::round(sum_b));
        }
    }
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
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
