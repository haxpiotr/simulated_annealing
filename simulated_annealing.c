#include "simulated_annealing.h"

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

double simulated_annealing(struct configuration* config,
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

    

    double current_result = function(x_0, N);

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
                    current_result = new_result;
                    copy_to_array(x_0, x_new, N);
                }
            }      
        }
        T = T * (1 - alpha);
    }

    return function(x_optimal, N);
}
