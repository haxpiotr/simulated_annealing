#include "simulated_annealing_pthread.h"
#include "simulated_annealing.h"
#include "functions.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

enum function_choice
{
    engval = 0,
    rosenbrock = 1,
    rastrigin = 2,
    quartic = 3,
    shannos = 4,
    unkown
};

int main(int argc, char *argv[])
{
    if(argc < 5)
    {
        printf("Call the application: sa_pthread <size_of_function> <threads> <test_iterations> <function> \n");
        return 0;
    }

    char* end_first = NULL;

    const unsigned int size_of_task = strtoul(argv[1],&end_first,10);

    char* end_second = NULL;

    const unsigned int threads = strtoul(argv[2],&end_second,10);

    char* end_third = NULL;

    const unsigned int test_iterations = strtoul(argv[3],&end_third,10);
    
    double x_left_boundaries[size_of_task];
    double x_right_boundaries[size_of_task];

    double x_0[size_of_task];
    double x_opt[size_of_task];

    struct configuration config = 
    {
        .initial_temperature = 1000,
        .alpha_coefficient = 0.01,
        .minimal_temperature = 0.00001,
        .iterations_per_temperature_step = 250
    };

    struct timeval st, et;

    int milliseconds_elapsed_pthreads = 0;

    double (*function)(double*,unsigned int);

    enum function_choice choice;

    if(strcmp("engval",argv[4])== 0)
    {
        function = engval_function;
        choice = engval;
    }
    else if(strcmp("rosenbrock",argv[4]) == 0)
    {
        function = generalized_rosenbrock_function;
        choice = rosenbrock;
    }
    else if(strcmp("rastrigin",argv[4]) == 0)
    {
        function = rastrigin_function;
        choice = rastrigin;
    }
    else if(strcmp("quartic",argv[4]) == 0)
    {
        function = quartic_function;
        choice = quartic;
    }
    else if(strcmp("shannos",argv[4]) == 0)
    {
        function = shannos_function;
        choice = shannos;
    }
    else
    {
        printf("Call the application: sa_pthread <size_of_function> <threads> <test_iterations> <function> \n");
        printf("<function>: engval, rosenbrock, rastrigin, quartic, shannos\n");
        return 0;
    }

    printf("Simulated Annealing using pthreads, size of %s function: %d run on %d threads out of %d available with %d test iterations\n"
    ,argv[4],size_of_task,threads,omp_get_num_procs(),test_iterations);

    for(unsigned int i = 0; i < test_iterations; ++i)
    {
        if(choice == engval)
        {
            engval_function_initialize(x_0,size_of_task);
            engval_function_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, (double)-10);
            set_range_to(x_right_boundaries, size_of_task, (double)10);
        }else if(choice == rosenbrock)
        {
            generalized_rosenbrock_initialize(x_0,size_of_task);
            generalized_rosenbrock_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, (double)-3);
            set_range_to(x_right_boundaries, size_of_task, (double)3);
        }else if(choice == rastrigin)
        {
            rastrigin_function_initialize(x_0,size_of_task);
            rastrigin_function_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, (double)-5.14);
            set_range_to(x_right_boundaries, size_of_task, (double)5.14);
        }else if(choice == quartic)
        {
            quartic_function_initialize(x_0,size_of_task);
            quartic_function_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, (double)-1.1);
            set_range_to(x_right_boundaries, size_of_task, (double)1.1);
        }else
        {
            shannos_function_initialize(x_0,size_of_task);
            shannos_function_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, (double)-1.2);
            set_range_to(x_right_boundaries, size_of_task, (double)1.2);
        }
        

        gettimeofday(&st,NULL);

        double result_pthreads = simulated_annealing_pthread(&config,
                                            function,
                                            size_of_task,
                                            x_0,
                                            x_opt,
                                            x_left_boundaries,
                                            x_right_boundaries,
                                            threads);

        gettimeofday(&et,NULL);

        int elapsed_in_pthreads = ((et.tv_sec - st.tv_sec) * 1000) + (et.tv_usec - st.tv_usec) / 1000;
        milliseconds_elapsed_pthreads += elapsed_in_pthreads;

        printf("Optimal solution for %s function using pthread\n", argv[4]);

        for(unsigned int i = 0; i < size_of_task; ++i)
        {
            printf("x[%d]: %.17g, ",i, x_opt[i]);
        }

        printf("\nSimulated Annealing using pthread, result: %.17g in %d ms\n", result_pthreads, elapsed_in_pthreads);
    }

    int avg_time_pthreads = milliseconds_elapsed_pthreads / test_iterations;

    printf("Simulated Annealing using pthread, average time of %d runs for %s function of size: %d: %d milliseconds\n",test_iterations,argv[4],size_of_task, avg_time_pthreads);

    int milliseconds_elapsed_seq = 0;

    int seed = time(NULL);

    for(unsigned int i = 0; i < test_iterations; ++i)
    {
        if(choice == engval)
        {
            engval_function_initialize(x_0,size_of_task);
            engval_function_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, (double)-10);
            set_range_to(x_right_boundaries, size_of_task, (double)10);
        }else if(choice == rosenbrock)
        {
            generalized_rosenbrock_initialize(x_0,size_of_task);
            generalized_rosenbrock_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, (double)-3);
            set_range_to(x_right_boundaries, size_of_task, (double)3);
        }else if(choice == rastrigin)
        {
            rastrigin_function_initialize(x_0,size_of_task);
            rastrigin_function_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, (double)-5.14);
            set_range_to(x_right_boundaries, size_of_task, (double)5.14);
        }else if(choice == quartic)
        {
            quartic_function_initialize(x_0,size_of_task);
            quartic_function_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, (double)-1.1);
            set_range_to(x_right_boundaries, size_of_task, (double)1.1);
        }else
        {
            shannos_function_initialize(x_0,size_of_task);
            shannos_function_initialize(x_opt,size_of_task);
            set_range_to(x_left_boundaries, size_of_task, (double)-1.2);
            set_range_to(x_right_boundaries, size_of_task, (double)1.2);
        }

        gettimeofday(&st,NULL);

        double result_seq = simulated_annealing(&config,
                                        function,
                                        size_of_task,
                                        x_0,
                                        x_opt,
                                        x_left_boundaries,
                                        x_right_boundaries,
                                        &seed);

        gettimeofday(&et,NULL);

        int elapsed_in_seq = ((et.tv_sec - st.tv_sec) * 1000) + (et.tv_usec - st.tv_usec) / 1000;
        milliseconds_elapsed_seq += elapsed_in_seq;

        printf("Optimal solution for %s function using sequential approach\n",argv[4]);

        for(unsigned int i = 0; i < size_of_task; ++i)
        {
            printf("x[%d]: %.17g, ",i, x_opt[i]);
        }

        printf("\nSimulated Annealing using sequential approach, result: %.17g in %d ms\n", result_seq, elapsed_in_seq);
    }

    int avg_time_seq = milliseconds_elapsed_seq / test_iterations;

    double pthreads_speedup = (double)avg_time_seq/(double)avg_time_pthreads;

    printf("Simulated Annealing speed up using pthreads with %d threads out of %d available: %f\n",threads, omp_get_num_procs(),pthreads_speedup);

    return 0;
}
