//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI + OpenMp + SIMD + Reordering Matrix Multiplication
//

#include <mpi.h>  // MPI Header
#include <omp.h> 
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include <stdio.h>
#include "matrix.hpp"

#define MASTER 0

int **init_matrix(int rows, int cols) {
    int *data = (int *)malloc(rows*cols*sizeof(int));
    int **array= (int **)malloc(rows*sizeof(int*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}


Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, size_t i_start, size_t i_end) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // In addition to OpenMP, SIMD, Memory Locality and Cache Missing,
    // Further Applying MPI
    // Note:
    // You can change the argument of the function 
    // for your convenience of task division
    #pragma omp parallel for
    for (size_t i = i_start; i < i_end; ++i) {
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
            "Invalid argument, should be: ./executable thread_num "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
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

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();


    size_t row_per_task = matrix1.getRows() / numtasks;
    size_t left_row = matrix1.getRows() % numtasks;
    std::vector<size_t> cuts(numtasks + 1, 0);
    size_t divide_left_row = 0;

    for(size_t i=0; i<numtasks; i++){
        if(divide_left_row < left_row){
            cuts[i+1] = cuts[i] + row_per_task + 1;
            divide_left_row++;
        }
        else {
            cuts[i+1] = cuts[i] + row_per_task;
        }
    }

    if (taskid == MASTER) {
        
        // Your Code Here for Synchronization!
        int **res;
        res = init_matrix(matrix1.getRows(), matrix2.getCols());
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, cuts[MASTER], cuts[MASTER+1]);

        for (int i = MASTER + 1; i < numtasks; i++) {
            int* start_row = res[cuts[i]];
            int length = (cuts[i+1] - cuts[i]) * matrix2.getCols();
            MPI_Recv(start_row, length, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            //printf("%d \n", res[cuts[i]][0]);
        }

        for(int i=cuts[MASTER+1]; i<matrix1.getRows(); i++){
            int *res_row = res[i];
            int *result_row = result[i];
            for(int j=0; j<matrix2.getCols(); j++){
                result_row[j] = res_row[j];
            }
            //printf("%d \n", result[i][1023]);
        }


        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;
    } 
    else {
        
        // Your Code Here for Synchronization!
        int **res;
        int length = (cuts[taskid+1]-cuts[taskid])*matrix2.getCols();
        res = init_matrix(length/matrix2.getCols(), matrix2.getCols());
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, cuts[taskid], cuts[taskid+1]);
        
        for(int i=cuts[taskid]; i<cuts[taskid+1]; i++){
            int *res_row = res[i-cuts[taskid]];
            int *result_row = result[i];
            for(int j=0; j<matrix2.getCols(); j++){
                res_row[j] = result_row[j];
            }
        }
        MPI_Send(&(res[0][0]), length, MPI_INT, MASTER, 0, MPI_COMM_WORLD);

    }

    MPI_Finalize();
    return 0;
}