#pragma once

struct configuration
{
    double initial_temperature;
    double alpha_coefficient;
    double minimal_temperature;
    unsigned int iterations_per_temperature_step;
};

double get_random();

double get_random_threadsafe(int* seed);

double get_new_point(double left_limit,
                     double right_limit,
                     double factor);

void fill_with_new_points(double* x,
                          unsigned int n,
                          double* left_limits,
                          double* right_limits,
                          double factor);


void fill_with_new_points_changing_factor(double* x,
                                          unsigned int n,
                                          double* left_limits,
                                          double* right_limits);

void fill_with_new_points_changing_factor_ts(double* x, unsigned int n, double* left_limits, double* right_limits, int* seed);

void copy_to_array(double* dest, double* source, unsigned int n);

void adjust_new_boundaries(double* left_limits,
                           double* right_limits,
                           unsigned int n, 
                           unsigned int processors,
                           unsigned int proc_index,
                           double* adjusted_left_limits,
                           double* adjusted_right_limits);