#include "simple_ml_openacc.hpp"
#include <openacc.h>
#include <math.h>
#include <cmath>
#include <algorithm> 

#pragma acc routine vector
void matrix_dot_openacc(const float *A, const float *B,
                        float *C, size_t M, size_t N, size_t K)
{
    // BEGIN YOUR CODE
    memset(C, 0, sizeof(float)*M*K);
    for (size_t k = 0; k < N; ++k) {
        const float* row2 = B + k*K;
        #pragma acc loop vector
        for (size_t i = 0; i < M; ++i) {
            float v1 = A[i*N+k]; 
            
            for (size_t j = 0; j < K; ++j) {
                C[i*K+j] += v1 * row2[j];
            }
        }
    }
    // END YOUR CODE
}

#pragma acc routine vector
void matrix_dot_trans_openacc(const float *A, const float *B, float *C, size_t N, size_t M, size_t K)
{
    // BEGIN YOUR CODE
    memset(C, 0, sizeof(float)*M*K);
    for (size_t k = 0; k < N; ++k) {
        const float* row2 = B + k*K;
        #pragma acc loop vector
        for (size_t i = 0; i < M; ++i) {
            float v1 = A[k*M+i]; 
            #pragma acc loop independent
            for (size_t j = 0; j < K; ++j) {
                C[i*K+j] += v1 * row2[j];
            }
        }
    }
    // END YOUR CODE
}


#pragma acc routine vector
void matrix_trans_dot_openacc(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
    // BEGIN YOUR CODE
    memset(C, 0, sizeof(float)*M*K);
    for (size_t k = 0; k < N; ++k) {
        for (size_t i = 0; i < M; ++i) {
            float v1 = A[i*N+k]; 
            #pragma acc loop vector
            for (size_t j = 0; j < K; ++j) {
                C[i*K+j] += v1 * B[j*N+k];
            }
        }
    }
    // END YOUR CODE
}

#pragma acc routine vector
void matrix_minus_openacc(float *A, const float *B, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc loop vector
    for(size_t i=0; i<m*n; i++){
        A[i] = A[i] - B[i];
     }
    // END YOUR CODE
}

#pragma acc routine vector
void matrix_mul_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc loop vector
    for(size_t i=0; i<m*n; i++){
        C[i] *= scalar;
    }
    // END YOUR CODE
}

#pragma acc routine vector
void matrix_div_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    float num = 1 / scalar;
    #pragma acc loop vector
    for(size_t i=0; i<m*n; i++){
        C[i] *= num;
    }
    // END YOUR CODE
}

#pragma acc routine vector
void matrix_softmax_normalize_openacc(float *C, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    float sum;
    float reversed_sum;
    #pragma acc loop vector
    for (size_t i = 0; i < m; i++) {
        sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += expf(C[i * n + j]);
        }
        reversed_sum = 1/sum;
        for (size_t j = 0; j < n; j++) {
            C[i * n + j] = expf(C[i * n + j]) * reversed_sum;
        }
    }
    // END YOUR CODE
}

#pragma acc routine vector
void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    memset(Y, 0, sizeof(float) * m * n);
    #pragma acc loop vector
    for (size_t i = 0; i < m; i++) {
        Y[i * n + y[i]] = 1;
    }
    // END YOUR CODE
}

#pragma acc routine worker
void softmax_regression_epoch_openacc(const float *X, const unsigned char *y,
                                      float *theta, size_t m, size_t n, size_t k,
                                      float lr, size_t batch, 
                                      float *Y_batch, float *logits, float *gradients)
{
    // BEGIN YOUR CODE
        #pragma acc loop worker
        for(size_t i=0; i<m; i+=batch){
            size_t M = std::min(i + batch, m);
            size_t size = M - i;
            const unsigned char *y_batch = y + i;
            const float *X_batch = X + i*n;

            matrix_dot_openacc(X_batch, theta, logits, size, n, k);
            matrix_softmax_normalize_openacc(logits, size, k);
            vector_to_one_hot_matrix_openacc(y_batch, Y_batch, size, k);
            matrix_minus_openacc(logits, Y_batch, size, k);
            matrix_dot_trans_openacc(X_batch, logits, gradients, size, n, k);
            matrix_mul_scalar_openacc(gradients, lr, n, k);
            matrix_div_scalar_openacc(gradients, size, n, k);
            matrix_minus_openacc(theta, gradients, n, k);
        }
    

    // END YOUR CODE
}

#pragma acc routine vector
float mean_softmax_loss_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes, float *log_vec, float *labels, float *h_y)
{
    // BEGIN YOUR CODE

    float sum;
    double loss = 0;
    #pragma acc loop independent
    for(size_t i=0; i<images_num; i++){
        sum = 0;
        size_t row = i*num_classes;
        
        #pragma acc loop vector reduction(+:sum)
        for(size_t j=0; j<num_classes; j++){
            sum += exp(result[row+j]);
        }
        log_vec[i] = log(sum);
    }

    vector_to_one_hot_matrix_openacc(labels_array, labels, images_num, num_classes);
    float sum1;
    #pragma acc loop independent
    for(size_t i=0; i<images_num; i++){
        sum1 = 0;
        size_t row = i*num_classes;

        #pragma acc loop vector reduction(+:sum1)
        for(size_t j=0; j<num_classes; j++){
            sum1 += result[row+j] * labels[row+j];  
        }
        h_y[i] = sum1;
    }
    matrix_minus_openacc(log_vec, h_y, 1, images_num);
    
    #pragma acc loop vector reduction(+:loss)
    for(size_t i=0; i<images_num; i++){
        loss += log_vec[i];
    }
    float mean_loss = loss / images_num;

    return mean_loss;

    // END YOUR CODE
}


void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
    float *Y_batch = new float[batch*num_classes];
    float *logits = new float[batch*num_classes];
    float *gradients = new float[train_data->input_dim*num_classes];
    float *log_vec = new float[train_data->images_num];
    float *labels = new float[train_data->images_num * num_classes];
    float *h_y = new float[train_data->images_num];
          
    #pragma acc data copyin(theta[0:size], \
                                train_data->images_matrix[0:train_data->images_num*train_data->input_dim], \
                                test_data->images_matrix[0:test_data->images_num*test_data->input_dim],   \
                                train_data->labels_array[0:train_data->images_num], \
                                test_data->labels_array[0:test_data->images_num]) \
                        create(train_result[0:size_tr], \
                                test_result[0:size_te], \
                                Y_batch[0:batch*num_classes], \
                                logits[0:batch*num_classes], \
                                gradients[0:train_data->input_dim*num_classes], \
                                log_vec[0:train_data->images_num], \
                                labels[0:train_data->images_num * num_classes], \
                                h_y[0:train_data->images_num])
    {
        auto start_time = std::chrono::high_resolution_clock::now();  
        for (size_t epoch = 0; epoch < epochs; epoch++){
            softmax_regression_epoch_openacc(train_data->images_matrix, train_data->labels_array, theta, train_data->images_num, train_data->input_dim, num_classes, lr, batch, Y_batch, logits, gradients);

            matrix_dot_openacc(train_data->images_matrix, theta, train_result, train_data->images_num, train_data->input_dim, num_classes);
            matrix_dot_openacc(test_data->images_matrix, theta, test_result, test_data->images_num, test_data->input_dim, num_classes);
            train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes, log_vec, labels, h_y);
            test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes, log_vec, labels, h_y);
            train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
            test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
            std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
        }            
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    }

  
   // END YOUR CODE
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}


#pragma acc routine vector
float mean_err_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float total_err = 0;
    float mean_error;
    float max_prob;
    size_t idx;
    size_t row;
    #pragma acc loop vector
    for(size_t i=0; i<images_num; i++){
        row = i*num_classes;
        max_prob = result[row];
        idx = 0;
        for(size_t j=1; j<num_classes; j++){
            if(result[row+j]>max_prob){
                max_prob = result[row+j];
                idx = j;
            }
        }
        if(idx!=labels_array[i]){
            total_err += 1;
        }
    }
    mean_error = total_err / images_num;
    return mean_error;

    // END YOUR CODE
}

#pragma acc routine vector
void matrix_mul_openacc(float *A, const float *B, size_t size)
{
    // BEGIN YOUR CODE
    #pragma acc loop vector
    for(size_t i=0; i<size; i++){
        A[i] *= B[i];
    }
    // END YOUR CODE
}

#pragma acc routine worker
void nn_epoch_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch, float *Z1, float *Z2, float *Y, float *G1, float *mask, float *W1_l, float *W2_l)
{
    // BEGIN YOUR CODE
    #pragma acc loop worker
    for (size_t i = 0; i < m; i += batch)
    {
        const float *X_b = X + i * n;
  
        matrix_dot_openacc(X_b, W1, Z1, batch, n, l);
        for(size_t j=0; j<batch*l; j++){
            if(Z1[j]<0){
                Z1[j] = 0;
            }
        }
        matrix_dot_openacc(Z1, W2, Z2, batch, l, k);
        matrix_softmax_normalize_openacc(Z2, batch, k);
        vector_to_one_hot_matrix_openacc(y+i, Y, batch, k);
        matrix_minus_openacc(Z2, Y, batch, k);
        matrix_trans_dot_openacc(Z2, W2, G1, batch, k, l);
        for(size_t j=0; j<batch*l; j++){
            if(Z1[j]>0){
                mask[j] = 1;
            }
            else{
                mask[j] = 0;
            }
        }
        matrix_mul_openacc(G1, mask, batch*l);
        matrix_dot_trans_openacc(X_b, G1, W1_l, batch, n, l);
        matrix_dot_trans_openacc(Z1, Z2, W2_l, batch, l, k);
        matrix_div_scalar_openacc(W1_l, batch, n, l);
        matrix_mul_scalar_openacc(W1_l, lr, n, l);
        matrix_div_scalar_openacc(W2_l, batch, l, k);
        matrix_mul_scalar_openacc(W2_l, lr, l, k);
        matrix_minus_openacc(W1, W1_l, n, l);
        matrix_minus_openacc(W2, W2_l, l, k);
    }
    // END YOUR CODE
}

void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_w1 = train_data->input_dim * hidden_dim;
    size_t size_w2 = hidden_dim * num_classes;
    float *W1 = new float[size_w1];
    float *W2 = new float[size_w2];
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(0.0, 1.0);
    for (size_t i = 0; i < size_w1; i++)
    {
        W1[i] = dist(rng);
    }
    for (size_t i = 0; i < size_w2; i++)
    {
        W2[i] = dist(rng);
    }
    matrix_div_scalar_openacc(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
    matrix_div_scalar_openacc(W2, sqrtf(num_classes), hidden_dim, num_classes);
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE

    float *log_vec = new float[train_data->images_num];
    float *labels = new float[train_data->images_num * num_classes];
    float *h_y = new float[train_data->images_num];
    float *Z1 = new float[batch * hidden_dim];
    float *Z2 = new float[batch * num_classes];
    float *Y = new float[batch * num_classes];
    float *G1 = new float[batch * hidden_dim];
    float *mask = new float[batch*hidden_dim];
    float *W1_l = new float[train_data->input_dim*hidden_dim];
    float *W2_l = new float[hidden_dim*num_classes];
              

    float *temp1 = new float[train_data->images_num * hidden_dim];
    float *temp2 = new float[test_data->images_num * hidden_dim];
    auto start_time = std::chrono::high_resolution_clock::now();
    #pragma acc data copyin(W1[0:size_w1], \
                            W2[0:size_w2], \
                                train_data->images_matrix[0:train_data->images_num*train_data->input_dim], \
                                test_data->images_matrix[0:test_data->images_num*test_data->input_dim],   \
                                train_data->labels_array[0:train_data->images_num], \
                                test_data->labels_array[0:test_data->images_num]) \
                        create(train_result[0:size_tr], \
                                test_result[0:size_te], \
                                log_vec[0:train_data->images_num], \
                                labels[0:train_data->images_num * num_classes], \
                                h_y[0:train_data->images_num], \
                                Z1[0:batch * hidden_dim], \
                                Z2[0:batch * num_classes], \
                                Y[0:batch * num_classes], \
                                G1[0:batch * hidden_dim], \
                                mask[0:batch*hidden_dim], \
                                W1_l[0:train_data->input_dim*hidden_dim], \
                                W2_l[0:hidden_dim*num_classes], \
                                temp1[0:train_data->images_num * hidden_dim], \
                                temp2[0:test_data->images_num * hidden_dim])
    {
        for (size_t epoch = 0; epoch < epochs; epoch++)
        {
            nn_epoch_openacc(train_data->images_matrix, train_data->labels_array, W1, W2, train_data->images_num, train_data->input_dim, hidden_dim, num_classes, lr, batch, Z1, Z2, Y, G1, mask, W1_l, W2_l);
            matrix_dot_openacc(train_data->images_matrix, W1, temp1, train_data->images_num, train_data->input_dim, hidden_dim);
            matrix_dot_openacc(test_data->images_matrix, W1, temp2, test_data->images_num, test_data->input_dim, hidden_dim);
            for(size_t i=0; i<train_data->images_num * hidden_dim; i++){
                if(temp1[i]<0){
                    temp1[i] = 0;
                }
            }
            matrix_dot_openacc(temp1, W2, train_result, train_data->images_num, hidden_dim, num_classes);
        
            for(size_t i=0; i<test_data->images_num * hidden_dim; i++){
                if(temp2[i]<0){
                    temp2[i] = 0;
                }
            }
            matrix_dot_openacc(temp2, W2, test_result, test_data->images_num, hidden_dim, num_classes);
            train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes, log_vec, labels, h_y);
            test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes, log_vec, labels, h_y);
            train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
            test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
            std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                    << std::fixed << std::setprecision(5) << train_loss << " |   "
                    << std::fixed << std::setprecision(5) << train_err << " |   "
                    << std::fixed << std::setprecision(5) << test_loss << " |  "
                    << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
        }
    }


    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
    // END YOUR CODE
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
