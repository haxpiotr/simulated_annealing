#pragma once
#include "random_helpers.h"

struct thread_arguments
{
    unsigned int thread_id;
    unsigned int iterations;
    unsigned int N;
    double* current_solution;
    double* best_solution;
    double* x_left_boundaries;
    double* x_right_boundaries;
    double (*function)(double*,unsigned int);
    double current_temperature;
};

void* simulated_annealing_pthread_part(void* args);

double simulated_annealing_pthread(struct configuration* config,
                                  double (*function)(double*,unsigned int),
                                  unsigned int N,
                                  double* x_0,
                                  double* x_optimal,
                                  double* x_left_boundaries,
                                  double* x_right_bondaries,
                                  unsigned int procs);


