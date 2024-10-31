#include <iostream>
#include <cmath>
#include <stdio.h>
#include <cuda_runtime.h> 

#include "utils.hpp"


__global__ void filter_pixel(unsigned char* input, unsigned char* output,
                          int width, int height, int num_channels)
{
    const int FILTER_SIZE = 3;
    double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
    };
    int h = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int w = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if(h < height-1 && w < width-1){
            
            unsigned char *r_value = &input[((h -1) * width + (w - 1)) * num_channels];
            unsigned char *g_value = r_value + 1;
            unsigned char *b_value = r_value + 2;

            output[(h * width + w) * 3] = static_cast<unsigned char>(std::round(
                *r_value * filter[0][0] + *(r_value + num_channels) * filter[0][1] + *(r_value + num_channels*2) * filter[0][2]
                + *(r_value + width * num_channels) * filter[1][0] + *(r_value + width * num_channels + num_channels) * filter[1][1] + *(r_value + width * num_channels + num_channels * 2) * filter[1][2]
                + *(r_value + width * 2 * num_channels) * filter[2][0] + *(r_value + width * 2 * num_channels + num_channels) * filter[2][1] + *(r_value + width * 2 * num_channels + num_channels * 2) * filter[2][2]));

            output[(h * width + w) * 3 + 1] = static_cast<unsigned char>(std::round(
                *g_value * filter[0][0] + *(g_value + num_channels) * filter[0][1] + *(g_value + num_channels*2) * filter[0][2]
                + *(g_value + width * num_channels) * filter[1][0] + *(g_value + width * num_channels + num_channels) * filter[1][1] + *(g_value + width * num_channels + num_channels * 2) * filter[1][2]
                + *(g_value + width * 2 * num_channels) * filter[2][0] + *(g_value + width * 2 * num_channels + num_channels) * filter[2][1] + *(g_value + width * 2 * num_channels + num_channels * 2) * filter[2][2]));

            output[(h * width + w) * 3 + 2] = static_cast<unsigned char>(std::round(
                *b_value * filter[0][0] + *(b_value + num_channels) * filter[0][1] + *(b_value + num_channels*2) * filter[0][2]
                + *(b_value + width * num_channels) * filter[1][0] + *(b_value + width * num_channels + num_channels) * filter[1][1] + *(b_value + width * num_channels + num_channels * 2) * filter[1][2]
                + *(b_value + width * 2 * num_channels) * filter[2][0] + *(b_value + width * 2 * num_channels + num_channels) * filter[2][1] + *(b_value + width * 2 * num_channels + num_channels * 2) * filter[2][2]));


        
    }
    
}


int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    // Allocate memory on host (CPU)
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, 
            input_jpeg.width * input_jpeg.height *
            input_jpeg.num_channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_output,
            input_jpeg.width * input_jpeg.height * 
            input_jpeg.num_channels * sizeof(unsigned char));
    cudaMemcpy(d_input, input_jpeg.buffer,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels *
                   sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 block(32, 32);
    dim3 grid((input_jpeg.height-2) / 32 + 1, (input_jpeg.width-2) / 32 + 1);
    //int blockSize = 64;
    //int numBlocks = (input_jpeg.height-2) / blockSize + 1;
    cudaEventRecord(start, 0);
    filter_pixel<<<grid, block>>>(d_input, d_output, input_jpeg.width,
                                        input_jpeg.height,
                                        input_jpeg.num_channels);

    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    cudaMemcpy(filteredImage, d_output,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory on device and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}