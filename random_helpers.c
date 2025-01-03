#include "random_helpers.h"

#include <stdlib.h>
#include <time.h>
#include <stdio.h>

double get_random()
{
    return (double)rand() / (double)RAND_MAX;
}

double get_random_threadsafe(int* seed)
{
    return (double)rand_r(seed) / (double)RAND_MAX;
}


double get_new_point(double left_limit, double right_limit, double factor)
{
    return left_limit + factor * (right_limit - left_limit);
}

void fill_with_new_points(double* x, unsigned int n, double* left_limits, double* right_limits, double factor)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        x[i] = get_new_point(left_limits[i], right_limits[i], factor);
    }
}

void fill_with_new_points_changing_factor(double* x, unsigned int n, double* left_limits, double* right_limits)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        const double factor = get_random();
        x[i] = get_new_point(left_limits[i], right_limits[i], factor);
    }
}

void fill_with_new_points_changing_factor_ts(double* x, unsigned int n, double* left_limits, double* right_limits, int* seed)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        const double factor = get_random_threadsafe(seed);
        x[i] = get_new_point(left_limits[i], right_limits[i], factor);
    }
}

void copy_to_array(double* dest, double* source, unsigned int n)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        dest[i] = source[i];
    }
}

void adjust_new_boundaries(double* left_limits,
                           double* right_limits,
                           unsigned int n, 
                           unsigned int processors,
                           unsigned int proc_index,
                           double* adjusted_left_limits,
                           double* adjusted_right_limits)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        double length = (right_limits[i] - left_limits[i]) / processors;
        adjusted_left_limits[i] = left_limits[i] + proc_index * length;
        adjusted_right_limits[i] = left_limits[i] + (proc_index + 1) * length;
    }
}