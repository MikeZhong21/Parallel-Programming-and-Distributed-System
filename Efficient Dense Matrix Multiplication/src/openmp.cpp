#include <immintrin.h>
#include <omp.h> 
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

Matrix matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const int* row2 = matrix2[k];
            __m256i val1 = _mm256_set1_epi32(matrix1[i][k]);
            int* res_row = result[i];
            
            if(N>=8){
                size_t j;
                for (j = 0; j < N; j+=8) {
                    __m256i result_row = _mm256_loadu_si256((__m256i*)(res_row + j));
                    __m256i row2_val = _mm256_loadu_si256((__m256i*)(row2 + j));
                    result_row = _mm256_add_epi32(result_row, _mm256_mullo_epi32(val1, row2_val));
                
                    __m128i result_low = _mm256_castsi256_si128(result_row);
                    __m128i result_high = _mm256_extracti128_si256(result_row, 1);
                    _mm_storeu_si128((__m128i*)(&res_row[j]), result_low);
                    _mm_storeu_si128((__m128i*)(&res_row[j+4]), result_high);
                }
                for (; j < N; ++j) {
                    res_row[j] += matrix1[i][k] * row2[j];
                }
            }
            else {
                for (size_t j = 0; j < N; ++j) {
                    res_row[j] += matrix1[i][k] * row2[j];
                }
            }

        }
    }


    return result;
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num"
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    const std::string matrix1_path = argv[2];
    const std::string matrix2_path = argv[3];
    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_openmp(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;
    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}