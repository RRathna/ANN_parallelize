# ANN_parallelize

mpicc -lm -o mpi_genann genann.c mpi_example.c
mpirun -n 4 ./mpi_genann
