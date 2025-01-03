#include "simulated_annealing.h"
#include "functions.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

int main()
{
    srand(time(NULL));

    double x_rastrigin_left_boundaries[] = {-2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2,
                                           -2,-2,-2,-2};

    double x_rastrigin_right_boundaries[] = {2,2,2,2,
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
        .alpha_coefficient = 0.02,
        .minimal_temperature = 0.00005,
        .iterations_per_temperature_step = 12000
    };

    double x_0_rastriginl_seq[] = {1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33};
    
    double x_opt_rastriginl_seq[] = {1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33};

    double x_0_rastriginl_omp[] = {1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33};
    
    double x_opt_rastriginl_omp[] = {1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33,
                                     1/33,1/33,1/33,1/33,1/33,1/33,1/33,1/33};

    const unsigned int size_of_task = sizeof(x_0_rastriginl_omp)/sizeof(double);

    printf("size_of_task: %d\n",size_of_task);

    struct timeval st, et;

    gettimeofday(&st,NULL);
    
    double result_rastrigin_seq = simulated_annealing(&config,
                                        rastrigin_function,
                                        size_of_task,
                                        x_0_rastriginl_seq,
                                        x_opt_rastriginl_seq,
                                        x_rastrigin_left_boundaries,
                                        x_rastrigin_right_boundaries);

    gettimeofday(&et,NULL);
    int elapsed = ((et.tv_sec - st.tv_sec) * 1000) + (et.tv_usec - st.tv_usec) / 1000;
    printf("simulated_annealing time: %d milliseconds\n",elapsed);
    printf("optimal solution for rastrigin function \n");

    for(unsigned int i = 0; i < size_of_task; ++i)
    {
         printf("x_opt_rosenbrock[%d]: %.17g\n",i, x_opt_rastriginl_seq[i]);
    }

    printf("result_rastrigin_seq: %.17g\n\n", result_rastrigin_seq);

    gettimeofday(&st,NULL);

    double result_rastrigin_omp = simulated_annealing_openmp(&config,
                                        rastrigin_function,
                                        size_of_task,
                                        x_0_rastriginl_omp,
                                        x_opt_rastriginl_omp,
                                        x_rastrigin_left_boundaries,
                                        x_rastrigin_right_boundaries);

    gettimeofday(&et,NULL);
    elapsed = ((et.tv_sec - st.tv_sec) * 1000) + (et.tv_usec - st.tv_usec) / 1000;
    printf("simulated_annealing_openmp time: %d milliseconds\n", elapsed);

    printf("optimal solution for rastrigin function using omp\n");


    for(unsigned int i = 0; i < size_of_task; ++i)
    {
         printf("x_opt_rastriginl_omp omp[%d]: %.17g\n",i, x_opt_rastriginl_omp[i]);
    }

    printf("result_rastrigin_omp: %.17g\n", result_rastrigin_omp);

    return 0;
}
