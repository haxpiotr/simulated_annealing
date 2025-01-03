#include "simulated_annealing.h"
#include "functions.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

int main()
{
    int seed = time(NULL);

    double x_engval_left_boundaries[] = {-2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2};

    double x_engval_right_boundaries[] = {2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2,
                                           2,2,2,2};

    struct configuration config = 
    {
        .initial_temperature = 1000,
        .alpha_coefficient = 0.01,
        .minimal_temperature = 0.00001,
        .iterations_per_temperature_step = 1200
    };

    double x_0_engval_seq[] = {1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65};
    
    double x_opt_engval_seq[] = {1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65,
                                     1/65,1/65,1/65,1/65,1/65,1/65,1/65,1/65};

    const unsigned int size_of_task = sizeof(x_0_engval_seq)/sizeof(double);

    printf("size_of_task: %d\n",size_of_task);

    struct timeval st, et;

    gettimeofday(&st,NULL);
    
    double result_engval_seq = simulated_annealing(&config,
                                        engval_function,
                                        size_of_task,
                                        x_0_engval_seq,
                                        x_opt_engval_seq,
                                        x_engval_left_boundaries,
                                        x_engval_right_boundaries,
                                        &seed);

    gettimeofday(&et,NULL);
    
    
    printf("optimal solution for rastrigin function \n");

    for(unsigned int i = 0; i < size_of_task; ++i)
    {
         printf("x_opt_engval_seq[%d]: %.17g\n",i, x_opt_engval_seq[i]);
    }

    printf("result_engval_seq: %.17g\n\n", result_engval_seq);

    int elapsed = ((et.tv_sec - st.tv_sec) * 1000) + (et.tv_usec - st.tv_usec) / 1000;
    printf("simulated_annealing_seq time for engval function of size: %d: %d milliseconds\n",size_of_task, elapsed);

    return 0;
}
