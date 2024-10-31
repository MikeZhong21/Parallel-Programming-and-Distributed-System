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

void oddEvenSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    /* Your code here!
       Implement parallel odd-even sort with MPI
    */
    bool sorted = false;


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
        
        int i;
        int begin;
        if(cuts[taskid]%2==0){
            i = 0;
            begin = 0;
        }
        else{
            i = 1;
            begin = 1;
        }
        for (; i < sub_vec.size() - 1; i += 2) {
            if (sub_vec[i] > sub_vec[i + 1]) {
                std::swap(sub_vec[i], sub_vec[i + 1]);
                sorted_task = false;
            }
        }
    
        if(taskid != numtasks-1 && i==sub_vec.size()-1){
            int num;
            MPI_Send(&sub_vec[i], 1, MPI_INT, taskid+1, 0, MPI_COMM_WORLD);
            MPI_Recv(&num, 1, MPI_INT, taskid+1, 1, MPI_COMM_WORLD, status);
            if(num < sub_vec[i]){
                sub_vec[i] = num;
                sorted_task = false;
            }
        }

        if(taskid != MASTER && begin==1){
            int temp;
            MPI_Recv(&temp, 1, MPI_INT, taskid-1, 0, MPI_COMM_WORLD, status);
            MPI_Send(&sub_vec[0], 1, MPI_INT, taskid-1, 1, MPI_COMM_WORLD);
            if(temp > sub_vec[0]){
                sub_vec[0] = temp;
                sorted_task = false;
            }
        }


        if(cuts[taskid]%2==0){
            i = 1;
            begin = 1;
        }
        else{
            i = 0;
            begin = 0;
        }
        for (; i < sub_vec.size() - 1; i += 2) {
            if (sub_vec[i] > sub_vec[i + 1]) {
                std::swap(sub_vec[i], sub_vec[i + 1]);
                sorted_task = false;
            }
        }
    

        if(taskid != numtasks-1 && i==sub_vec.size()-1){
            int num;
            MPI_Send(&sub_vec[i], 1, MPI_INT, taskid+1, 3, MPI_COMM_WORLD);
            MPI_Recv(&num, 1, MPI_INT, taskid+1, 4, MPI_COMM_WORLD, status);
            if(num < sub_vec[i]){
                sub_vec[i] = num;
                sorted_task = false;
            }
        }

        if(taskid != MASTER && begin==1){
            int temp;
            MPI_Recv(&temp, 1, MPI_INT, taskid-1, 3, MPI_COMM_WORLD, status);
            MPI_Send(&sub_vec[0], 1, MPI_INT, taskid-1, 4, MPI_COMM_WORLD);
            if(temp > sub_vec[0]){
                sub_vec[0] = temp;
                sorted_task = false;
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