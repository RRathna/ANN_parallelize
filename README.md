# ANN_parallelize


Instructions to run MPI version

  1. mpicc -lm -o mpi_exe genann.c mpi_example.c
  2. mpirun -n 4 ./mpi_exe

Instructions to run OMP version

  1. gcc -lm -fopenmp -o omp_exe omp_genann.c omp_example.c
  2. export OMP_NUM_THREADS=4
  3. ./omp_exe

