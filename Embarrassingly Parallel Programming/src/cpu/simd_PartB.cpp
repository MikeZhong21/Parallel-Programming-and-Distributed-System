#include <iostream>
#include <chrono>
#include <stdio.h>
#include <cmath>
#include <immintrin.h>

#include "../utils.hpp"

const int FILTER_SIZE = 3;
const float filter[FILTER_SIZE][FILTER_SIZE] = {
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
    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;


    // Mask used for shuffling when store int32s to u_int8 array

    __m128i shuffle_r_low = _mm_setr_epi8(0, -1, -1, 4, 
                                    -1, -1, 8, -1, 
                                    -1, 12, -1, -1, 
                                    -1, -1, -1, -1);
    __m128i shuffle_g_low = _mm_setr_epi8(-1, 0, -1, -1, 
                                    4, -1, -1, 8, 
                                    -1, -1, 12, -1, 
                                    -1, -1, -1, -1);
    __m128i shuffle_b_low = _mm_setr_epi8(-1, -1, 0, -1, 
                                    -1, 4, -1, -1, 
                                    8, -1, -1, 12, 
                                    -1, -1, -1, -1);

    __m128i shuffle_r_high1 = _mm_setr_epi8(-1, -1, -1, -1, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1, 
                                    0, -1, -1, 4);
    __m128i shuffle_g_high1 = _mm_setr_epi8(-1, -1, -1, -1, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1, 
                                    -1, 0, -1, -1);
    __m128i shuffle_b_high1 = _mm_setr_epi8(-1, -1, -1, -1, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1, 
                                    -1, -1, 0, -1);

    __m128i shuffle_r_high2 = _mm_setr_epi8(-1, -1, 8, -1, 
                                    -1, 12, -1, -1, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1);
    __m128i shuffle_g_high2 = _mm_setr_epi8(4, -1, -1, 8, 
                                    -1, -1, 12, -1, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1);
    __m128i shuffle_b_high2 = _mm_setr_epi8(-1, 4, -1, -1, 
                                    8, -1, -1, 12, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1);

    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height + 16];

    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    auto start_time = std::chrono::high_resolution_clock::now();    // Start recording time
    for (int height = 1; height < input_jpeg.height-1; height++) {
        for (int width = 1; width < input_jpeg.width - 8; width+=8){
            __m256 sum_red = _mm256_setzero_ps();
            __m256 sum_green = _mm256_setzero_ps();
            __m256 sum_blue = _mm256_setzero_ps();
            for (int i = -1; i <=  1; i++) {
                for (int j = -1; j <= 1; j++){
                    int index = ((height + i) * input_jpeg.width + (width + j)) * input_jpeg.num_channels;
                    __m256 filter_weight = _mm256_set1_ps(filter[i+1][j+1]);
                    
                    /*__m256i indices_red = _mm256_setr_epi32(index, index + 3, index + 6, index + 9,
                                                     index + 12, index + 15, index + 18, index + 21);
                    __m256i indices_green = _mm256_setr_epi32(index+1, index + 4, index + 7, index + 10,
                                                     index + 13, index + 16, index + 19, index + 22);
                    __m256i indices_blue = _mm256_setr_epi32(index+2, index + 5, index + 8, index + 11,
                                                     index + 14, index + 17, index + 20, index + 23);     
                    __m256 red = _mm256_cvtepi32_ps(_mm256_i32gather_epi32(
                        reinterpret_cast<int*>(input_jpeg.buffer), indices_red, sizeof(unsigned char)));
                    __m256 green = _mm256_cvtepi32_ps(_mm256_i32gather_epi32(
                        reinterpret_cast<int*>(input_jpeg.buffer), indices_green, sizeof(unsigned char)));
                    __m256 blue = _mm256_cvtepi32_ps(_mm256_i32gather_epi32(
                        reinterpret_cast<int*>(input_jpeg.buffer), indices_blue, sizeof(unsigned char)));
                    */

                    __m128i red_chars = _mm_loadu_si128((__m128i*) (reds + (height + i) * input_jpeg.width + (width + j)));
                    __m256i red_ints = _mm256_cvtepu8_epi32(red_chars);
                    __m256 red = _mm256_cvtepi32_ps(red_ints);

                    __m128i green_chars = _mm_loadu_si128((__m128i*) (greens + (height + i) * input_jpeg.width + (width + j)));
                    __m256i green_ints = _mm256_cvtepu8_epi32(green_chars);
                    __m256 green = _mm256_cvtepi32_ps(green_ints);

                    __m128i blue_chars = _mm_loadu_si128((__m128i*) (blues+(height + i) * input_jpeg.width + (width + j)));
                    __m256i blue_ints = _mm256_cvtepu8_epi32(blue_chars);
                    __m256 blue = _mm256_cvtepi32_ps(blue_ints);

                    __m256 red_results = _mm256_mul_ps(red, filter_weight);
                    __m256 green_results = _mm256_mul_ps(green, filter_weight);
                    __m256 blue_results = _mm256_mul_ps(blue, filter_weight);

                    sum_red = _mm256_add_ps(sum_red, red_results);
                    sum_green = _mm256_add_ps(sum_green, green_results);
                    sum_blue = _mm256_add_ps(sum_blue, blue_results);
                    
                }       
            }
            __m256i sum_r = _mm256_cvtps_epi32(sum_red);
            __m256i sum_g = _mm256_cvtps_epi32(sum_green);
            __m256i sum_b = _mm256_cvtps_epi32(sum_blue);

            __m128i r_low = _mm256_castsi256_si128(sum_r);
            __m128i r_high = _mm256_extracti128_si256(sum_r, 1);

            __m128i g_low = _mm256_castsi256_si128(sum_g);
            __m128i g_high = _mm256_extracti128_si256(sum_g, 1);

            __m128i b_low = _mm256_castsi256_si128(sum_b);
            __m128i b_high = _mm256_extracti128_si256(sum_b, 1); 

            __m128i pix1_r_low = _mm_shuffle_epi8(r_low, shuffle_r_low);
            __m128i pix1_g_low = _mm_shuffle_epi8(g_low, shuffle_g_low);
            __m128i pix1_b_low = _mm_shuffle_epi8(b_low, shuffle_b_low);
            __m128i pix1_low = _mm_add_epi8(_mm_add_epi8(pix1_r_low, pix1_g_low), pix1_b_low);

            __m128i pix1_r_high = _mm_shuffle_epi8(r_high, shuffle_r_high1);
            __m128i pix1_g_high = _mm_shuffle_epi8(g_high, shuffle_g_high1);
            __m128i pix1_b_high = _mm_shuffle_epi8(b_high, shuffle_b_high1);
            __m128i pix1_high = _mm_add_epi8(_mm_add_epi8(pix1_r_high, pix1_g_high), pix1_b_high);

            __m128i pix1 = _mm_add_epi8(pix1_low, pix1_high);

            __m128i pix2_r = _mm_shuffle_epi8(r_high, shuffle_r_high2);
            __m128i pix2_g = _mm_shuffle_epi8(g_high, shuffle_g_high2);
            __m128i pix2_b = _mm_shuffle_epi8(b_high, shuffle_b_high2);
            __m128i pix2 = _mm_add_epi8(_mm_add_epi8(pix2_r, pix2_g), pix2_b);

            _mm_storeu_si128((__m128i*)(&filteredImage[ (height * input_jpeg.width + width) * input_jpeg.num_channels]), pix1);
            _mm_storeu_si128((__m128i*)(&filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels+16]), pix2);

        }
        int remain = (input_jpeg.width - 2) % 8;
        for(int w = input_jpeg.width-remain-1; w < input_jpeg.width - 1; w++){
            filteredImage[(height * input_jpeg.width + w) * input_jpeg.num_channels]
                = static_cast<unsigned char>(std::round(
                reds[((height - 1) * input_jpeg.width + (w - 1))] * filter[0][0] 
                + reds[((height - 1) * input_jpeg.width + (w - 1)) + 1] * filter[0][1] 
                + reds[((height - 1) * input_jpeg.width + (w - 1)) + 2]  * filter[0][2]
                + reds[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width]  * filter[1][0] 
                + reds[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width + 1] * filter[1][1] 
                + reds[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width + 2] * filter[1][2]
                + reds[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width * 2] * filter[2][0] 
                + reds[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width * 2 + 1] * filter[2][1] 
                + reds[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width * 2 + 2] * filter[2][2]));

            filteredImage[(height * input_jpeg.width + w) * input_jpeg.num_channels + 1]
                = static_cast<unsigned char>(std::round(
                greens[((height - 1) * input_jpeg.width + (w - 1))] * filter[0][0] 
                + greens[((height - 1) * input_jpeg.width + (w - 1)) + 1] * filter[0][1] 
                + greens[((height - 1) * input_jpeg.width + (w - 1)) + 2]  * filter[0][2]
                + greens[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width]  * filter[1][0] 
                + greens[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width + 1] * filter[1][1] 
                + greens[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width + 2] * filter[1][2]
                + greens[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width * 2] * filter[2][0] 
                + greens[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width * 2 + 1] * filter[2][1] 
                + greens[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width * 2 + 2] * filter[2][2]));

            filteredImage[(height * input_jpeg.width + w) * input_jpeg.num_channels + 2]
                = static_cast<unsigned char>(std::round(
                blues[((height - 1) * input_jpeg.width + (w - 1))] * filter[0][0] 
                + blues[((height - 1) * input_jpeg.width + (w - 1)) + 1] * filter[0][1] 
                + blues[((height - 1) * input_jpeg.width + (w - 1)) + 2]  * filter[0][2]
                + blues[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width]  * filter[1][0] 
                + blues[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width + 1] * filter[1][1] 
                + blues[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width + 2] * filter[1][2]
                + blues[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width * 2] * filter[2][0] 
                + blues[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width * 2 + 1] * filter[2][1] 
                + blues[((height - 1) * input_jpeg.width + (w - 1)) + input_jpeg.width * 2 + 2] * filter[2][2]));
    
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();  // Stop recording time
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output Gray JPEG Image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    
    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
} 