#include "simulated_annealing_openmpi.h"
#include "simulated_annealing.h"
#include "functions.h"

#include <mpi.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

enum function_choice
{
    engval = 0,
    rosenbrock = 1,
    rastrigin = 2,
    quartic = 3,
    shannos = 4,
    unkown
};

int main(int argc, char** argv) 
{
    MPI_Init(NULL, NULL);      // initialize MPI environment

    if(argc < 3)
    {
        printf("Usage: mpirun -np <processors> ./sa_openmpi <size of function> <function(rastrigin|quartic)>\n");
        MPI_Finalize();
        return 0; 
    }

    char* end_first = NULL;

    const unsigned int size_of_task = strtoul(argv[1],&end_first,10);
    
    double x_left_boundaries[size_of_task];
    double x_right_boundaries[size_of_task];

    struct configuration config = 
    {
        .initial_temperature = 1000,
        .alpha_coefficient = 0.01,
        .minimal_temperature = 0.00001,
        .iterations_per_temperature_step = 250
    };

    double x_0[size_of_task];
    double x_opt[size_of_task];

    double (*function)(double*,unsigned int);

    if(strcmp("rastrigin",argv[2]) == 0)
    {
        function = rastrigin_function;
        rastrigin_function_initialize(x_0,size_of_task);
        rastrigin_function_initialize(x_opt,size_of_task);
        set_range_to(x_left_boundaries, size_of_task, -5.14);
        set_range_to(x_right_boundaries, size_of_task, 5.14);
    }
    else if(strcmp("quartic",argv[2]) == 0)
    {
        function = quartic_function;
        quartic_function_initialize(x_0,size_of_task);
        quartic_function_initialize(x_opt,size_of_task);
        set_range_to(x_left_boundaries, size_of_task, -1.1);
        set_range_to(x_right_boundaries, size_of_task, 1.1);
    }
    else
    {
        printf("Usage: mpirun -np <processors> ./sa_openmpi <size of function> <function(rastrigin|quartic)>\n");
        printf("Usage: specify function as rastrigin or quartic\n");
        MPI_Finalize();
        return 0;
    }
    
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    unsigned int iterations = config.iterations_per_temperature_step / world_size;

    struct configuration new_config = 
    {
        .initial_temperature = 1000,
        .alpha_coefficient = 0.01,
        .minimal_temperature = 0.00001,
        .iterations_per_temperature_step = iterations
    };

    int seed = time(NULL);

    struct timeval st, et;

    if(world_rank == 0)
    {
        gettimeofday(&st,NULL);
    }

    double y = simulated_annealing_openmpi(&new_config,
                                   function,
                                   size_of_task,
                                   x_0,
                                   x_opt,
                                   x_left_boundaries,
                                   x_right_boundaries,
                                   &seed);

    MPI_Barrier(MPI_COMM_WORLD);

    if(world_rank == 0)
    {
        gettimeofday(&et,NULL);

        int elapsed_openmpi = ((et.tv_sec - st.tv_sec) * 1000) + (et.tv_usec - st.tv_usec) / 1000;

        printf("## Simulated Annealing OpenMPI ## time for %s function of size: %d: %d milliseconds on rank: %d\n",argv[2],size_of_task, elapsed_openmpi, world_rank);
        printf("## Simulated Annealing OpenMPI ## result: %f:\n", y);

        if(strcmp("rastrigin",argv[2]) == 0)
        {
            function = rastrigin_function;
            rastrigin_function_initialize(x_0,size_of_task);
            rastrigin_function_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, -5.14);
            set_range_to(x_right_boundaries, size_of_task, 5.14);
        }
        else if(strcmp("quartic",argv[2]) == 0)
        {
            function = quartic_function;
            quartic_function_initialize(x_0,size_of_task);
            quartic_function_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, -1.1);
            set_range_to(x_right_boundaries, size_of_task, 1.1);
        }
        else
        {
            printf("Usage: mpirun -np <processors> ./sa_openmpi <size of function> <function(rastrigin|quartic)>\n");
            printf("Usage: specify function as rastrigin or quartic\n");
            MPI_Finalize();
            return 0;
        }

        int seed_seq = time(NULL);

        gettimeofday(&st,NULL);

        double y_seq = simulated_annealing(&config,
                                    function,
                                    size_of_task,
                                    x_0,
                                    x_opt,
                                    x_left_boundaries,
                                    x_right_boundaries,
                                    &seed_seq);

        gettimeofday(&et,NULL);

        int elapsed_sequential = ((et.tv_sec - st.tv_sec) * 1000) + (et.tv_usec - st.tv_usec) / 1000;

        printf("## Simulated annealing sequential time for %s function of size: %d: %d milliseconds\n",argv[2], size_of_task, elapsed_sequential);

        printf("## Simulated annealing sequential result: %f:\n", y_seq);

        double speedup = (double)elapsed_sequential/(double)elapsed_openmpi;

        printf("## Simulated annealing speedup for %s function of size %u using OpenMPI: %f:\n",argv[2], size_of_task, speedup);
    }

    MPI_Finalize(); // finish MPI environment

    return 0;
}
