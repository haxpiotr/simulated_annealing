#pragma once

#include "random_helpers.h"

double simulated_annealing(struct configuration* config,
                           double (*function)(double*,unsigned int),
                           unsigned int N,
                           double* x_0,
                           double* x_optimal,
                           double* x_left_boundaries,
                           double* x_right_bondaries,
                           int* seed);
