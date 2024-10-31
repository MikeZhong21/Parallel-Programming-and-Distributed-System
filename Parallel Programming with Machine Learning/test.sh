#!/bin/bash

mkdir -p build
cd src

# simple
g++ simple_ml_ext.cpp softmax_classifier.cpp -Ofast -o ../build/softmax
pgc++ -acc simple_ml_ext.cpp simple_ml_openacc.cpp softmax_classifier_openacc.cpp -o ../build/softmax_openacc

# nn
g++ simple_ml_ext.cpp simple_ml_openacc.cpp nn_classifier.cpp -Ofast -o ../build/nn
pgc++ -acc -ta=tesla:cc75 -Mfma -O3 simple_ml_ext.cpp simple_ml_openacc.cpp nn_classifier_openacc.cpp  -o ../build/nn_openacc

rm *.o

cd ..

sbatch sbatch.sh
