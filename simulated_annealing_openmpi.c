#include "simulated_annealing_openmpi.h"

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

double simulated_annealing_openmpi(struct configuration* config,
                           double (*function)(double*,unsigned int),
                           unsigned int N,
                           double* x_0,
                           double* x_optimal,
                           double* x_left_boundaries,
                           double* x_right_bondaries,
                           int* seed)
{
    const double eps = config->minimal_temperature;
    double T = config->initial_temperature;
    const unsigned int L = config->iterations_per_temperature_step;
    const double alpha = config->alpha_coefficient;

    int name_len;
    
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);

    int world_rank; // the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size; // number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double current_result = function(x_0,N);

    printf("## Simulated Annealing OpenMPI from processor %s, rank %d out of %d processors, initial result: %.17g ##\n",
            processor_name, world_rank, world_size, current_result);

    while(T > eps)
    {
        for(unsigned int k = 0; k < L ; ++k)
        {
            double x_new[N] = {};
            fill_with_new_points_changing_factor_ts(x_new, N, x_left_boundaries, x_right_bondaries,seed);
            
            const double new_result = function(x_new, N);
                        
            if(new_result < current_result)
            {
                current_result = new_result;
                copy_to_array(x_0, x_new, N);
                copy_to_array(x_optimal, x_new, N);
            }
            else
            {
                const double r = get_random_threadsafe(seed);
                const double E = exp( (current_result - new_result) / T );
                if (r < E)
                {
                    copy_to_array(x_0, x_new, N);
                }
            }
        }

        for(unsigned int rank = 0 ; rank < world_size; ++rank)
        {
            if(rank == world_rank)
            {
                continue;
            }
            
            MPI_Send(&current_result, 1, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD);
        }

        for(unsigned int rank = 0 ; rank < world_size; ++rank)
        {
            if(rank == world_rank)
            {
                continue;
            }

            double received_result;
            MPI_Recv(&received_result, 1, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                
            if(received_result < current_result)
            {
                current_result = received_result;
            }
        }

        T = T * (1 - alpha);
    }


    if(world_rank != 0)
    {
        unsigned int main_rank = 0;
        MPI_Request req;
        MPI_Isend(x_optimal, N, MPI_DOUBLE, main_rank, 1, MPI_COMM_WORLD,&req);
    }
    else
    {
        for(unsigned int rank = 0 ; rank < world_size; ++rank)
        {
            if(rank == world_rank)
            {
                continue;
            }

            double received_opt[N];
            MPI_Recv(&received_opt, N, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                
            double received_result = function(received_opt,N);
            if(received_result < current_result)
            {
                copy_to_array(x_optimal, received_opt, N);
            }
        }
    }
    
    return function(x_optimal, N);
}
