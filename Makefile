CC=gcc
CFLAGS=-I. -O3 -march=native
CFLAGS_OPENMP=-I. -fopenmp -O3 -march=native
DEPS = functions.h random_helpers.h simulated_annealing.h simulated_annealing_openmp.h
OBJ_SEQ = functions.o random_helpers.o simulated_annealing.o main_sequential.o
OBJ_OPENMP = functions.o random_helpers.o simulated_annealing.o simulated_annealing_openmp.o main_openmp.o
LIBS=-lm

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS_OPENMP)

sa_sequential: $(OBJ_SEQ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

sa_openmp: $(OBJ_OPENMP)
	$(CC) -o $@ $^ $(CFLAGS_OPENMP) $(LIBS)

sa_openmpi:
	mpicc random_helpers.c functions.c simulated_annealing.c simulated_annealing_openmpi.c main_openmpi.c -o sa_openmpi -I. -lm -O3 -march=native

sa_upc:
	/home/piotr/upc/bin/upcc main_upc2.c -o sa_upc -I. -lm -network=smp

clean:
	rm -f sa_openmp
	rm -f sa_sequential
	rm -f sa_openmpi
	rm -f sa_upc
	rm -f *.o
