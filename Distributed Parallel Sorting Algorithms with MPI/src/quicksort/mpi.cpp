//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Quick Sort with MPI
//


#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <mpi.h>
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

std::vector<int> merge(std::vector<int>& recv, std::vector<int>& vec){
    int len_recv = recv.size();
    int len_vec = vec.size();
    std::vector<int> temp;
    temp.resize(len_recv+len_vec);
    int idx_recv = 0;
    int idx_vec = 0;
    int idx = 0;
    while(idx_recv<len_recv && idx_vec<len_vec){
        if(recv[idx_recv] <= vec[idx_vec]){
            temp[idx++] = recv[idx_recv++];
        }
        else if(recv[idx_recv] > vec[idx_vec]){
            temp[idx++] = vec[idx_vec++];
        }
    }
    while(idx_recv<len_recv){
        temp[idx++] = recv[idx_recv++];
    }
    while(idx_vec<len_vec){
        temp[idx++] = vec[idx_vec++];
    }
    return temp;
}

void quickSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    /* Your code here!
       Implement parallel quick sort with MPI
    */
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
    
    if(taskid==MASTER){
        for(int i=1; i<numtasks; i++){
            MPI_Send(&vec[cuts[i]], cuts[i+1]-cuts[i], MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        int low = cuts[MASTER];
        int high = cuts[MASTER+1]-1;
        std::stack<int> sort_stk;
        sort_stk.push(low);
        sort_stk.push(high);

        while(!sort_stk.empty()){
            high = sort_stk.top();
            sort_stk.pop();
            low = sort_stk.top();
            sort_stk.pop();
            int pivot = partition(vec, low, high);
            if(low < pivot-1){
                sort_stk.push(low);
                sort_stk.push(pivot-1);
            }
            if(pivot+1 < high){
                sort_stk.push(pivot+1);
                sort_stk.push(high);
            }
        }
        std::stack<int>().swap(sort_stk);
        vec.resize(cuts[MASTER+1]-cuts[MASTER]);
        std::vector<int> recv_vec;
        std::queue<std::vector<int>> merge_queue;
        merge_queue.push(vec);
        for(int i=MASTER+1; i<numtasks; i++){
            
            recv_vec.resize(cuts[i+1]-cuts[i]);
            MPI_Recv(&recv_vec[0], cuts[i+1]-cuts[i], MPI_INT, i, 1, MPI_COMM_WORLD, status);
            merge_queue.push(recv_vec);
        }

        while(merge_queue.size() > 1){
            std::vector<int> vec1 = merge_queue.front();
            merge_queue.pop();
            std::vector<int> vec2 = merge_queue.front();
            merge_queue.pop();
            merge_queue.push(merge(vec1, vec2));
        }
        vec = merge_queue.front();
    }


    else{
        std::vector<int> temp_vec;
        temp_vec.resize(cuts[taskid+1]-cuts[taskid]);
        MPI_Recv(&temp_vec[0], cuts[taskid+1]-cuts[taskid], MPI_INT, MASTER, 0, MPI_COMM_WORLD, status);
  
        int low = 0;
        int high = temp_vec.size()-1;
        std::stack<int> sort_stk;
        sort_stk.push(low);
        sort_stk.push(high);

        while(!sort_stk.empty()){
            high = sort_stk.top();
            sort_stk.pop();
            low = sort_stk.top();
            sort_stk.pop();
            int pivot = partition(temp_vec, low, high);
            if(low < pivot-1){
                sort_stk.push(low);
                sort_stk.push(pivot-1);
            }
            if(pivot+1 < high){
                sort_stk.push(pivot+1);
                sort_stk.push(high);
            }
        }
        std::stack<int>().swap(sort_stk);
        MPI_Send(&temp_vec[0], cuts[taskid+1]-cuts[taskid], MPI_INT, MASTER, 1, MPI_COMM_WORLD);
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

    quickSort(vec, numtasks, taskid, &status);
    if (taskid == MASTER) {

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Quick Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}