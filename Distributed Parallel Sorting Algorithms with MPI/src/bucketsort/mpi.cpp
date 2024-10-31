//most recent
//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include <stdio.h>
#include "../utils.hpp"

#define MASTER 0

int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }

    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

void quickSort(std::vector<int> &vec, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(vec, low, high);
        quickSort(vec, low, pivotIndex - 1);
        quickSort(vec, pivotIndex + 1, high);
    }
}


void bucketSort(std::vector<int>& vec, int num_buckets, int numtasks, int taskid, MPI_Status* status) {
    /* Your code here!
       Implement parallel bucket sort with MPI
    */

    int buckets_per_task = num_buckets / numtasks;
    int left_buckets = num_buckets % numtasks;
    std::vector<int> cuts(numtasks+1, 0);
    int divide_left_buckets = 0;
    for(size_t i=0; i<numtasks; i++){
        if(divide_left_buckets<left_buckets){
            cuts[i+1] = cuts[i] + buckets_per_task + 1;
            divide_left_buckets++;
        }
        else {
            cuts[i+1] = cuts[i] + buckets_per_task;
        }
    }
    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());

    int range = max_val - min_val + 1;
    int range_per_task = range / numtasks;

    if (taskid==MASTER){
        std::vector<int> vect;
        vect.resize(vec.size());

        int task_min = min_val;
        int task_max = min_val + range_per_task; 
        int task_range = task_max - task_min + 1;
        int task_num_buckets = cuts[MASTER+1] - cuts[MASTER];

        int small_bucket_size = task_range / task_num_buckets;
        int large_bucket_size = small_bucket_size + 1;
        int large_bucket_num = range - small_bucket_size * task_num_buckets;
        int boundary = task_min + large_bucket_num * large_bucket_size;

        std::vector<std::vector<int>> buckets(task_num_buckets);
    
        for (std::vector<int>& bucket : buckets) {
            bucket.reserve(large_bucket_size);
        }

        for (int num : vec) {
            if(num>=task_min && num<task_max){
                int index;
                if (num < boundary) {
                    index = (num - task_min) / large_bucket_size;
                } 
                else if(num < task_max && num >= boundary){
                    index = large_bucket_num + (num - boundary) / small_bucket_size;
                }
                if (index >= task_num_buckets) {
                    // Handle elements at the upper bound
                    index = task_num_buckets - 1;
                }
                buckets[index].push_back(num);
            
            }
            
        }

        for (std::vector<int>& bucket : buckets) {
            quickSort(bucket, 0, bucket.size()-1);
        }

        int index = 0;
        for (const std::vector<int>& bucket : buckets) {
            for (int num : bucket) {
                vect[index++] = num;
            }
        }
        //printf("%d\n", index);
        for(int i=MASTER+1; i<numtasks; i++){
    
            int length;

            MPI_Recv(&length, 1, MPI_INT, i, 0, MPI_COMM_WORLD, status);
            std::vector<int> temp;
            temp.resize(length);
            
            MPI_Recv(&temp[0], length, MPI_INT, i, 1, MPI_COMM_WORLD, status);
            //printf("%d\n", index);
            for(int j=0; j<length; j++){
                vect[index++] = temp[j];
            }
        }
        vec = vect;
    }

    else{
        std::vector<int> vect;
        int vec_len = 0;

        int task_min = min_val + taskid *  range_per_task;
        int task_max;
        int task_range;
        if(taskid != numtasks-1){
            task_max = min_val + (taskid + 1) * range_per_task;
        }
        else{
            task_max = max_val + 1;
        } 
        task_range = task_max - task_min + 1;
        int task_num_buckets = cuts[taskid+1] - cuts[taskid];

        int small_bucket_size = task_range / task_num_buckets;
        int large_bucket_size = small_bucket_size + 1;
        int large_bucket_num = range - small_bucket_size * task_num_buckets;
        int boundary = task_min + large_bucket_num * large_bucket_size;

        std::vector<std::vector<int>> buckets(task_num_buckets);
    
        for (std::vector<int>& bucket : buckets) {
            bucket.reserve(large_bucket_size);
        }

        for (int num : vec) {
            if(num>=task_min && num<task_max){
                int index;
                if (num < boundary && num >= task_min) {
                    index = (num - task_min) / large_bucket_size;
                } 
                else if(num < task_max && num >= boundary){
                    index = large_bucket_num + (num - boundary) / small_bucket_size;
                }
                if (index >= task_num_buckets) {
                    // Handle elements at the upper bound
                    index = task_num_buckets - 1;
                }
                    buckets[index].push_back(num);
                    vec_len++;
            }
        }

        for (std::vector<int>& bucket : buckets) {
            quickSort(bucket, 0, bucket.size()-1);
        }
    
        int index = 0;
        vect.resize(vec_len);
        
        for (const std::vector<int>& bucket : buckets) {
            for (int num : bucket) {
                vect[index++] = num;
            }
        }
        
        MPI_Send(&vec_len, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        MPI_Send(&vect[0], vec_len, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
        
    }
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size bucket_num\n"
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

    const int bucket_num = atoi(argv[2]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    bucketSort(vec, bucket_num, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}