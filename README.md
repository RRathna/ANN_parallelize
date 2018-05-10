# ANN_parallelize


Instructions to run MPI version

  mpicc -lm -o mpi_genann genann.c mpi_example.c
  mpirun -n 4 ./mpi_genann
