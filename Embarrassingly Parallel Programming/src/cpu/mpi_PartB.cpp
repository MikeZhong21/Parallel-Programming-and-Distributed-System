#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <mpi.h>    // MPI Header

#include "utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

const int FILTER_SIZE = 3;
double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    const char * input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int height_per_task = (input_jpeg.height-2) / numtasks;
    int remain_height = (input_jpeg.height-2) % numtasks;

    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_height = 0;

    for (int i = 0; i < numtasks; i++) {
        if (divided_left_height < remain_height) {
            cuts[i+1] = cuts[i] + height_per_task + 1;
            divided_left_height++;
        } 
        else{
           cuts[i+1] = cuts[i] + height_per_task; 
        } 
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (taskid == MASTER) {
        auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
        for (int h = cuts[MASTER]+1; h < cuts[MASTER + 1]+1; h++) {
            for(int w = 1; w < input_jpeg.width-1; w++){
                unsigned char *r_value = &input_jpeg.buffer[((h - 1) * input_jpeg.width + (w - 1)) * input_jpeg.num_channels];
                unsigned char *g_value = r_value + 1;
                unsigned char *b_value = r_value + 2;
            
            filteredImage[(h * input_jpeg.width + w) * 3] = static_cast<unsigned char>(std::round(
                *r_value * filter[0][0] + *(r_value + input_jpeg.num_channels) * filter[0][1] + *(r_value + input_jpeg.num_channels*2) * filter[0][2]
                + *(r_value + input_jpeg.width * input_jpeg.num_channels) * filter[1][0] + *(r_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels) * filter[1][1] + *(r_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[1][2]
                + *(r_value + input_jpeg.width * 2 * input_jpeg.num_channels) * filter[2][0] + *(r_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels) * filter[2][1] + *(r_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[2][2]));
            
            filteredImage[(h * input_jpeg.width + w) * 3 + 1] = static_cast<unsigned char>(std::round(
                *g_value * filter[0][0] + *(g_value + input_jpeg.num_channels) * filter[0][1] + *(g_value + input_jpeg.num_channels*2) * filter[0][2]
                + *(g_value + input_jpeg.width * input_jpeg.num_channels) * filter[1][0] + *(g_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels) * filter[1][1] + *(g_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[1][2]
                + *(g_value + input_jpeg.width * 2 * input_jpeg.num_channels) * filter[2][0] + *(g_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels) * filter[2][1] + *(g_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[2][2]));
            
            filteredImage[(h * input_jpeg.width + w) * 3 + 2] = static_cast<unsigned char>(std::round(
                *b_value * filter[0][0] + *(b_value + input_jpeg.num_channels) * filter[0][1] + *(b_value + input_jpeg.num_channels*2) * filter[0][2]
                + *(b_value + input_jpeg.width * input_jpeg.num_channels) * filter[1][0] + *(b_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels) * filter[1][1] + *(b_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[1][2]
                + *(b_value + input_jpeg.width * 2 * input_jpeg.num_channels) * filter[2][0] + *(b_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels) * filter[2][1] + *(b_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[2][2]));
            
            }
        }

        for (int i = MASTER + 1; i < numtasks; i++) {
            unsigned char* start_pos = filteredImage + (cuts[i]+1)*input_jpeg.width * input_jpeg.num_channels;
            int length = cuts[i+1] - cuts[i];
            MPI_Recv(start_pos, length * input_jpeg.width * input_jpeg.num_channels, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

                const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG to file\n";
            MPI_Finalize();
            return -1;
        }

        // Release the memory
        delete[] input_jpeg.buffer;
        delete[] filteredImage;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    }

    else{
        int length = cuts[taskid + 1] - cuts[taskid];
        auto filteredImage = new unsigned char[length * input_jpeg.width * input_jpeg.num_channels];
        for(int h = cuts[taskid]+1; h < cuts[taskid + 1]+1; h++){
            for(int w = 1; w < input_jpeg.width-1; w++){
                unsigned char *r_value = &input_jpeg.buffer[((h - 1) * input_jpeg.width + (w - 1)) * input_jpeg.num_channels];
                unsigned char *g_value = r_value + 1;
                unsigned char *b_value = r_value + 2;
            
            int h1 = h - cuts[taskid] - 1;

            filteredImage[(h1 * input_jpeg.width + w) * 3] = static_cast<unsigned char>(std::round(
                *r_value * filter[0][0] + *(r_value + input_jpeg.num_channels) * filter[0][1] + *(r_value + input_jpeg.num_channels*2) * filter[0][2]
                + *(r_value + input_jpeg.width * input_jpeg.num_channels) * filter[1][0] + *(r_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels) * filter[1][1] + *(r_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[1][2]
                + *(r_value + input_jpeg.width * 2 * input_jpeg.num_channels) * filter[2][0] + *(r_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels) * filter[2][1] + *(r_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[2][2]));
            
            filteredImage[(h1 * input_jpeg.width + w) * 3 + 1] = static_cast<unsigned char>(std::round(
                *g_value * filter[0][0] + *(g_value + input_jpeg.num_channels) * filter[0][1] + *(g_value + input_jpeg.num_channels*2) * filter[0][2]
                + *(g_value + input_jpeg.width * input_jpeg.num_channels) * filter[1][0] + *(g_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels) * filter[1][1] + *(g_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[1][2]
                + *(g_value + input_jpeg.width * 2 * input_jpeg.num_channels) * filter[2][0] + *(g_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels) * filter[2][1] + *(g_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[2][2]));
            
            filteredImage[(h1 * input_jpeg.width + w) * 3 + 2] = static_cast<unsigned char>(std::round(
                *b_value * filter[0][0] + *(b_value + input_jpeg.num_channels) * filter[0][1] + *(b_value + input_jpeg.num_channels*2) * filter[0][2]
                + *(b_value + input_jpeg.width * input_jpeg.num_channels) * filter[1][0] + *(b_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels) * filter[1][1] + *(b_value + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[1][2]
                + *(b_value + input_jpeg.width * 2 * input_jpeg.num_channels) * filter[2][0] + *(b_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels) * filter[2][1] + *(b_value + input_jpeg.width * 2 * input_jpeg.num_channels + input_jpeg.num_channels * 2) * filter[2][2]));
            
            }
        }

        MPI_Send(filteredImage, length * input_jpeg.width * input_jpeg.num_channels, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        
        delete[] filteredImage;
    }  

    MPI_Finalize();
    return 0;
}


