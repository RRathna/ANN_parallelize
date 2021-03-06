# ANN_parallelize
This project parallelizes the popular serial library for ANN, genann - https://codeplea.com/genann

The training data used is mnist. This code uses mnist loader from Nuri Park's project - https://github.com/projectgalateia/mnist

MPI version - Very fast, gradual drop in accuracy (due to averaging weights between nodes after each iterations). Uses the library as is and runs the training distributedly. The additions are all made to MPI_example.c
OMP version - Work in progress - give lesser accuracy, optimal at 8 threads. Changed the train function in the library to make use of shared memory parallelism. (see omp_genann.c and omp_example.c)

You can use make command to get the executables for each of the versions or follow the instructions below:

Instructions to run the original version

  1. gcc -lm -o exe genann.c example.c
  2. ./exe

Instructions to run MPI version

  1. mpicc -lm -o mpi_exe genann.c mpi_example.c
  2. mpirun -n 4 ./mpi_exe

Instructions to run OMP version

  1. gcc -lm -fopenmp -o omp_exe omp_genann.c omp_example.c
  2. export OMP_NUM_THREADS=4
  3. ./omp_exe

