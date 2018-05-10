CCFLAGS = -fopenmp -Wall -Wshadow -O2 -g
LDLIBS = -lm


all: example omp_example mpi_example

example: example.o genann.o

omp_example: omp_example.o omp_genann.o

mpi_example: mpi_example.o genann.o


clean:
	$(RM) *.o
	$(RM) *.exe
	$(RM) persist.txt
