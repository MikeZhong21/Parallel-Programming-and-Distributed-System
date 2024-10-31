#include <iostream>
#include <chrono>
#include <cmath>
#include <pthread.h>
#include "utils.hpp"

const int FILTER_SIZE = 3;
double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};
unsigned char *filteredImage;
unsigned char *buffer;

struct ThreadData {
    int width;
    int start;
    int end;
};

void* filter_pixel(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    for(int h = data->start; h < data->end; h++){
        for(int w = 1; w < data->width-1; w++){

        unsigned char *r_value = &buffer[((h - 1) * data->width + (w - 1)) * 3];
        unsigned char *g_value = r_value + 1;
        unsigned char *b_value = r_value + 2;
            
            filteredImage[(h * data->width + w) * 3]
                = static_cast<unsigned char>(std::round(
                *r_value * filter[0][0] + *(r_value + 3) * filter[0][1] + *(r_value + 3*2) * filter[0][2]
                + *(r_value + data->width * 3) * filter[1][0] + *(r_value + data->width * 3 + 3) * filter[1][1] + *(r_value + data->width * 3 + 3 * 2) * filter[1][2]
                + *(r_value + data->width * 2 * 3) * filter[2][0] + *(r_value + data->width * 2 * 3 + 3) * filter[2][1] + *(r_value + data->width * 2 * 3 + 3 * 2) * filter[2][2]));

            filteredImage[(h * data->width + w) * 3 + 1]
                = static_cast<unsigned char>(std::round(
                *g_value * filter[0][0] + *(g_value + 3) * filter[0][1] + *(g_value + 3*2) * filter[0][2]
                + *(g_value + data->width * 3) * filter[1][0] + *(g_value + data->width * 3 + 3) * filter[1][1] + *(g_value + data->width * 3 + 3 * 2) * filter[1][2]
                + *(g_value + data->width * 2 * 3) * filter[2][0] + *(g_value + data->width * 2 * 3 + 3) * filter[2][1] + *(g_value + data->width * 2 * 3 + 3 * 2) * filter[2][2]));

            filteredImage[(h * data->width + w) * 3 + 2]
                = static_cast<unsigned char>(std::round(
                *b_value * filter[0][0] + *(b_value + 3) * filter[0][1] + *(b_value + 3*2) * filter[0][2]
                + *(b_value + data->width * 3) * filter[1][0] + *(b_value + data->width * 3 + 3) * filter[1][1] + *(b_value + data->width * 3 + 3 * 2) * filter[1][2]
                + *(b_value + data->width * 2 * 3) * filter[2][0] + *(b_value + data->width * 2 * 3 + 3) * filter[2][1] + *(b_value + data->width * 2 * 3 + 3 * 2) * filter[2][2]));

            
        }
    }
    return NULL;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_threads\n";
        return -1;
    }

    int num_threads = std::stoi(argv[3]); 

    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);

    filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;

    buffer = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; i++)
    {
        buffer[i] = input_jpeg.buffer[i];
    }

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int size = input_jpeg.height / num_threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].width = input_jpeg.width;
        thread_data[i].start = i * size + 1;
        if(i==num_threads-1){
            thread_data[i].end = input_jpeg.height-1;
        }
        else{
            thread_data[i].end = (i+1)*size + 1;
        }
        pthread_create(&threads[i], NULL, filter_pixel, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
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