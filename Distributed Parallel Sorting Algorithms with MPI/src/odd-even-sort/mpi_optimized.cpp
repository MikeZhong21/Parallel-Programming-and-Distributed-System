//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Odd-Even Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include <stdio.h>
#include "../utils.hpp"

#define MASTER 0

void merge(std::vector<int>& vec, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary vectors
    std::vector<int> L(n1);
    std::vector<int> R(n2);

    // Copy data to temporary vectors L[] and R[]
    for (int i = 0; i < n1; i++) {
        L[i] = vec[l + i];
    }
    for (int i = 0; i < n2; i++) {
        R[i] = vec[m + 1 + i];
    }

    // Merge the temporary vectors back into v[l..r]
    int i = 0; // Initial index of the first subarray
    int j = 0; // Initial index of the second subarray
    int k = l; // Initial index of the merged subarray

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            vec[k] = L[i];
            i++;
        } else {
            vec[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        vec[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        vec[k] = R[j];
        j++;
        k++;
    }
}

// Main function to perform merge sort on a vector v[]
void mergeSort(std::vector<int>& vec, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        // Sort first and second halves
        mergeSort(vec, l, m);
        mergeSort(vec, m + 1, r);

        // Merge the sorted halves
        merge(vec, l, m, r);
    }
}


void oddEvenSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    /* Your code here!
       Implement parallel odd-even sort with MPI
    */
    bool sorted = false;
    if(taskid==MASTER && numtasks==1){

        while (!sorted) {
            sorted = true;

            // Perform the odd phase
            for (int i = 1; i < vec.size() - 1; i += 2) {
                if (vec[i] > vec[i + 1]) {
                    std::swap(vec[i], vec[i + 1]);
                    sorted = false;
                }
            }

            // Perform the even phase
            for (int i = 0; i < vec.size() - 1; i += 2) {
                if (vec[i] > vec[i + 1]) {
                    std::swap(vec[i], vec[i + 1]);
                    sorted = false;
                }
            }
        }
        return;
    }

    size_t size = vec.size();
    int idx_per_task = size / numtasks;
    int left_idx = size % numtasks;
    std::vector<size_t> cuts(numtasks + 1, 0);
    size_t divide_left_idx = 0;

    for(size_t i=0; i<numtasks; i++){
        if(divide_left_idx < left_idx){
            cuts[i+1] = cuts[i] + idx_per_task + 1;
            divide_left_idx++;
        }
        else {
            cuts[i+1] = cuts[i] + idx_per_task;
        }
    }

    bool sorted_task = false;
    std::vector<int> sub_vec;
    int start = cuts[taskid];
    int end = cuts[taskid+1];

    //printf("start: %d\n", start);
    //printf("end: %d\n", end);

    sub_vec.resize(end-start);
    for(int i=start; i<end; i++){
        sub_vec[i-start] = vec[i];
    }

    while(!sorted){
        sorted_task = true;
        if(taskid % 2 == 0 && taskid != numtasks-1){
            MPI_Send(&sub_vec[0], end-start, MPI_INT, taskid+1, 0, MPI_COMM_WORLD);
            MPI_Recv(&sub_vec[0], end-start, MPI_INT, taskid+1, 1, MPI_COMM_WORLD, status);
        }
        else if(taskid % 2 == 1){
            std::vector<int> temp;
            temp.resize(cuts[taskid]-cuts[taskid-1]);
            MPI_Recv(&temp[0], cuts[taskid]-cuts[taskid-1], MPI_INT, taskid-1, 0, MPI_COMM_WORLD, status);
            
            std::vector<int> combinedArray;
            combinedArray.reserve(sub_vec.size() + temp.size());
            combinedArray.insert(combinedArray.end(), temp.begin(), temp.end());
            combinedArray.insert(combinedArray.end(), sub_vec.begin(), sub_vec.end());

            //int size = combinedArray.size();
            //printf("combined size: %d\n", size);

            for (size_t i = 0; i < combinedArray.size()-1; ++i) {
                if (combinedArray[i] > combinedArray[i + 1]) {
                    sorted_task = false; 
                    break;
                }
            }
            mergeSort(combinedArray, 0, combinedArray.size()-1);

            MPI_Send(&combinedArray[0], cuts[taskid]-cuts[taskid-1], MPI_INT, taskid-1, 1, MPI_COMM_WORLD);
            for(int i=0; i<cuts[taskid+1]-cuts[taskid]; i++){
                sub_vec[i] = combinedArray[i+cuts[taskid]-cuts[taskid-1]];
            }
        }

        if(taskid % 2 == 1 && taskid != numtasks-1){
            MPI_Send(&sub_vec[0], end-start, MPI_INT, taskid+1, 0, MPI_COMM_WORLD);
            MPI_Recv(&sub_vec[0], end-start, MPI_INT, taskid+1, 1, MPI_COMM_WORLD, status);
        }
        else if(taskid!=MASTER && taskid % 2 == 0){
            std::vector<int> temp;
            temp.resize(cuts[taskid]-cuts[taskid-1]);
            MPI_Recv(&temp[0], cuts[taskid]-cuts[taskid-1], MPI_INT, taskid-1, 0, MPI_COMM_WORLD, status);
            std::vector<int> combinedArray;
            combinedArray.reserve(sub_vec.size() + temp.size());
            combinedArray.insert(combinedArray.end(), temp.begin(), temp.end());
            combinedArray.insert(combinedArray.end(), sub_vec.begin(), sub_vec.end());

            for (size_t i = 0; i < combinedArray.size()-1; ++i) {
                if (combinedArray[i] > combinedArray[i + 1]) {
                    sorted_task = false; 
                    break;
                }
            }

            mergeSort(combinedArray, 0, combinedArray.size()-1);
            MPI_Send(&combinedArray[0], cuts[taskid]-cuts[taskid-1], MPI_INT, taskid-1, 1, MPI_COMM_WORLD);
            for(int i=0; i<cuts[taskid+1]-cuts[taskid]; i++){
                sub_vec[i] = combinedArray[i+cuts[taskid]-cuts[taskid-1]];
            }
        }
        MPI_Allreduce(&sorted_task, &sorted, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    }
    if(taskid!=MASTER){
        MPI_Send(&sub_vec[0], end-start, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
    }
    
    if(taskid==MASTER){
        for(int i=0; i<cuts[MASTER+1]; i++){
            vec[i] = sub_vec[i];
        }
        std::vector<int> sub;
        for(int i=MASTER+1; i<numtasks; i++){
            sub.resize(cuts[i+1]-cuts[i]);
            MPI_Recv(&sub[0], cuts[i+1]-cuts[i], MPI_INT, i, 0, MPI_COMM_WORLD, status);
            for(int j=cuts[i]; j<cuts[i+1]; j++){
                vec[j] = sub[j-cuts[i]];
            }
        }
    }
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
            );
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

    const int size = atoi(argv[1]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    oddEvenSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Odd-Even Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}