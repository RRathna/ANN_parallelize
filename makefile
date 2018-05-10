
all: exe omp_exe mpi_exe

exe: example.o genann.o
	gcc -lm -o exe genann.c example.c

omp_example: omp_example.o omp_genann.o
	gcc -lm -fopenmp -o omp_exe omp_genann.c omp_example.c

mpi_exe: mpi_example.o genann.o
	mpicc -lm -fopenmp -o mpi_exe genann.c mpi_example.c


clean:
	$(RM) *.o
	$(RM) *.exe
	$(RM) persist.txt
