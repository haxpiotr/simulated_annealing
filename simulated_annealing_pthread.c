#include "simulated_annealing_pthread.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

void* simulated_annealing_pthread_part(void* args)
{
    struct thread_arguments* arguments = (struct thread_arguments*)args;

    double T = arguments->current_temperature;
    unsigned int L = arguments->iterations;
    unsigned int thread_id = arguments->thread_id;
    double* initial_solution = arguments->current_solution;
    double* optimal_solution = arguments->best_solution;
    double* x_left_boundaries = arguments->x_left_boundaries;
    double* x_right_boundaries = arguments->x_right_boundaries;
    unsigned int N = arguments->N;
    double (*function)(double*,unsigned int);
    function = arguments->function;

    int first_seed = time(NULL);
    int seed_thread = first_seed + thread_id;

    double current_solution_local[N];
    double best_solution_local[N];

    copy_to_array(current_solution_local,initial_solution,N);
    copy_to_array(best_solution_local,initial_solution,N);

    double current_result = function(current_solution_local, N);

    for(unsigned int k = 0; k < L; ++k)
    {   
        double x_new[N] = {};              
        fill_with_new_points_changing_factor_ts(x_new,N, x_left_boundaries, x_right_boundaries, &seed_thread);
        const double new_result = function(x_new, N);
        
        if(new_result < current_result)
        {
            current_result = new_result;
            copy_to_array(current_solution_local, x_new, N);
            copy_to_array(best_solution_local, x_new, N); 
        }
        else
        {
            const double r = get_random_threadsafe(&seed_thread);
            const double E = exp( (current_result - new_result) / T );
            if (r < E)
            {
                current_result = new_result;
                copy_to_array(current_solution_local, x_new, N);
            }
        }
    }

    copy_to_array(optimal_solution, best_solution_local, N);
}

double simulated_annealing_pthread(struct configuration* config,
                                  double (*function)(double*,unsigned int),
                                  unsigned int N,
                                  double* x_0,
                                  double* x_optimal,
                                  double* x_left_boundaries,
                                  double* x_right_bondaries,
                                  unsigned int procs)
{
    const double eps = config->minimal_temperature;
    const unsigned int L = config->iterations_per_temperature_step;
    const double alpha = config->alpha_coefficient;

    copy_to_array(x_optimal,x_0,N);
        
    double current_solution[procs][N];
    double best_solution[procs][N];
    double current_result[procs];

    for(unsigned int i = 0; i < procs; ++i)
    {
        copy_to_array(current_solution[i],x_0,N);
        copy_to_array(best_solution[i],x_0,N);
        current_result[i] = function(x_0,N);
    }
         
    double T = config->initial_temperature;

    struct thread_arguments arguments_for_pthread[procs];

    for(unsigned int i = 0; i < procs; ++i)
    {
        arguments_for_pthread[i].thread_id = i;
        arguments_for_pthread[i].iterations = L/procs;
        arguments_for_pthread[i].N = N;
        arguments_for_pthread[i].current_solution = current_solution[i];
        arguments_for_pthread[i].best_solution = best_solution[i];
        arguments_for_pthread[i].x_left_boundaries = x_left_boundaries;
        arguments_for_pthread[i].x_right_boundaries = x_right_bondaries;
        arguments_for_pthread[i].function = function;
        arguments_for_pthread[i].current_temperature = T;
    }
    
    while(T > eps)
    {            
        pthread_t threads[procs];

        for(unsigned int i = 0; i < procs; ++i)
        {
            pthread_create(&threads[i], NULL, simulated_annealing_pthread_part, &arguments_for_pthread[i]);
        }
        for(unsigned int i = 0; i < procs; ++i)
        {
            pthread_join(threads[i], NULL);
        }
        
        for(unsigned int i = 1; i < procs ;++i)
        {
            if(function(best_solution[i-1],N) < function(best_solution[i],N))
            {
                copy_to_array(best_solution[i],best_solution[i-1],N);
                copy_to_array(current_solution[i],best_solution[i-1],N);
            }
        }

        T = T * (1 - alpha);

        for(unsigned int i = 0; i < procs; ++i)
        {
            arguments_for_pthread[i].current_temperature = T;
        }
    }

    double best_result = function(best_solution[0], N);
    for(unsigned int i = 1; i < procs; ++i)
    {
        const double temp_best_result = function(best_solution[i], N);
        if(temp_best_result < best_result)
        {
            best_result = temp_best_result;
            copy_to_array(x_optimal,best_solution[i],N);
        }
    }
    
    return best_result;
}

