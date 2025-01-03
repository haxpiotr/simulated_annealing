#include "simulated_annealing_openmp.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

double simulated_annealing_openmp(struct configuration* config,
                                  double (*function)(double*,unsigned int),
                                  unsigned int N,
                                  double* x_0,
                                  double* x_optimal,
                                  double* x_left_boundaries,
                                  double* x_right_bondaries)
{
    const double eps = config->minimal_temperature;
    const unsigned int L = config->iterations_per_temperature_step;
    const double alpha = config->alpha_coefficient;

    copy_to_array(x_optimal,x_0,N);

    const unsigned int procs = omp_get_num_procs();
        
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

    int first_seed = time(NULL);
    
    while(T > eps)
    {            
        #pragma omp parallel for
        for(unsigned int k = 0; k < L; ++k)
        {   
            const unsigned int executor = omp_get_thread_num();
            int seed_omp = first_seed + executor;

            double x_new[N] = {};              
            fill_with_new_points_changing_factor_ts(x_new,N, x_left_boundaries, x_right_bondaries, &seed_omp);
            const double new_result = function(x_new, N);
            
            if(new_result < current_result[executor])
            {
                current_result[executor] = new_result;
                copy_to_array(current_solution[executor], x_new, N);
                copy_to_array(best_solution[executor], x_new, N); 
            }
            else
            {
                const double r = get_random_threadsafe(&seed_omp);
                const double E = exp( (current_result[executor] - new_result) / T );
                if (r < E)
                {
                    current_result[executor] = new_result;
                    copy_to_array(current_solution[executor], x_new, N);
                }
            }
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

