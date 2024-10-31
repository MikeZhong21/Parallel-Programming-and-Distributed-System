# Parallel Programming Projects

## Project 1:  Embarrassingly Parallel Programming
#### Part A
Transfer an image from RGB to grayscale. 
It utilizes the point operation (a function is applied to every pixel in an image or in a selection). 
The key point is that the function operates only on the pixel’s current value, which makes it completely embarrassingly parallel.
In this project, we use NTSC formula to be the function applied to the RGB image. <br/>
<p align="center"> $Gray = 0.299 * Red + 0.587 * Green + 0.114 * Blue$ <p/>

#### Part B
Apply the simplest size-3 low-pass filter with equal weights to smooth the input JPEG image. <br/>  
Image Filtering involves applying a function to every pixel in an image or selection but where the
function utilizes not only the pixels current value but the value of neighboring pixels.
we have a filter matrix of size 3x3, and we slide that filter matrix across the image to compute the
filtered value by element-wise multipling and summation, and optimize the computation performance using 6 parallel programming techniques.

## Project 2:  Efficient Dense Matrix Multiplication
Apply parallel programming techniques to optimize the performance of dense matrix multiplication by considering factors such as memory locality, SIMD (Single
Instruction, Multiple Data), thread-level parallelism, and process-level parallelism.
#### Memory Locality
* Reorder the nested loop
* Avoid cache misses
#### SIMD
* Handle 8 elements at a time：<br/>
load 8 same elements from matrix 1 to _m256i register and load 8 consecutive elements from matrix 2 to _m256i register
* The program loads the row of result matrix to store the calculation result in the innermost loop.
#### Thread-level Parallelism

#### Process-Level Parallelism

## Project 3: Distributed Parallel Sorting Algorithms with MPI
Optimize the performance of classic sorting algorithms.
#### Quick Sort using MPI 
* The program divides the whole vector into equal parts (sub vectors as equal as 
possible for the equal workload of each process). The size of the vector is 
determined, and each task calculates the index range it needs to sort.
* In each process, the program sorts the sub vector that is assigned by the master 
process using iterative quick sort.
* Finally, the master process will combine the sorted sub vectors from each 
processes using merging (the idea is generated from merge sort) so that the vector is 
in the right order. 
#### Bucket Sort using MPI
* The program calculates the number of buckets 
assigned to each process (buckets_per_task) and the number of remaining buckets 
that need to be distributed among the processes (left_buckets).
* The “cuts” vector stores the boundary index of buckets that each process is 
responsible for. 
* Each process will calculate the number range it is responsible for sorting. 
* In each process, the elements from the input vector “vec” within the task's range 
are distributed into their respective buckets based on the range and bucket sizes. 
The elements are sorted within each bucket using the quickSort function.
* Finally, iterate all the buckets and the elements inside each bucket, store the sorted 
elements in the vector and send the vector to the master process.
* Master process simply combines the receiving vector from each process and form 
the sorted vector.

#### Odd Even Sort using MPI

#### Odd Even Sort (merge sort version)
#### Taditional Odd Even Sort VS. Odd Even Merge Sort
* Traditional approach treats each single number as a sorting and swapping 
unit.
* Odd even merge sort treats all the elements in the process as a sorting unit.
* Instead of swapping single elements each time, the new approach uses 
merge sort in each odd and even phase to sort all the elements in the 
neighboring processes.
* Even phase
even number of processes will send its sub vector to its next 
neighboring odd process, and odd process will do the merge sort 
and send back the first part of the vector back to the even process.
* Odd phase
odd number of processes will send its sub vector to its next 
neighboring even process, and even process will do the merge sort 
and send back the first part of the vector back to the odd process.

## Project 4: Parallel Programming with Machine Learning

#### Train MNIST with softmax regression
#### Accelerate softmax with OpenACC
#### Train MNIST with neural network
#### Accelerate neural network with OpenACC

## Parallel Programming Languages Used
* SIMD
* MPI 
* Pthread
* OpenMP
* CUDA
* OpenACC
