#include "functions.h"

#include <math.h>
#include <stdio.h>

void set_range_to(double* x, unsigned int n, double val)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        x[i] = val;
    }
}

double quadratic_function(double* x, unsigned int n)
{
    double result = 0;

    for(unsigned int i = 2; i < n; ++i)
    {
        result += 100 * (pow(x[i], 2) + pow(x[i-1], 2)) + pow(x[i-2], 2);
    }

    return result;
}

double shannos_function(double* x, unsigned int n)
{
    double initial = pow(2 * x[0] -1, 2);
    double additional = 0;

    for(unsigned int i = 1; i < n; ++i)
    {
        additional += i * pow(2 * x[i-1] - x[i], 2);
    }
    
    return additional + initial;
}

void shannos_function_initialize(double* x, unsigned int n)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        if(i%2)
        {
            x[i] = -1;
        }else
        {
            x[i] = 1;
        }
    }
}

double generalized_rosenbrock_function(double* x, unsigned int n)
{
    double result = 0;

    for(unsigned int i = 1; i < n; ++i)
    {
        result += 100 * (pow( x[i] - pow( x[i-1], 2), 2)) + pow(1 - x[i-1], 2);
    }

    return 1 + result;
}

void generalized_rosenbrock_initialize(double* x, unsigned int n)
{
    double val = (double)1/(double)(n+1);

    for(unsigned int i = 0; i < n; ++i)
    {
        x[i] = 3 - val;
    }
}

double rastrigin_function(double* x, unsigned int n)
{
    double result = 0;

    for(unsigned int i = 0; i < n; ++i)
    {
        result += pow(x[i],2) - 10 * cos(2 * M_PI * x[i]);
    }

    return 10 * n + result;
}

void rastrigin_function_initialize(double* x, unsigned int n)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        x[i] = 4.52;
    }
}

double engval_function(double* x, unsigned int n)
{
    double result = 0;

    for(unsigned int i = 1; i < n; ++i)
    {
        result += pow( (pow( x[i-1], 2) + pow( x[i], 2)) , 2) - 4 * x[i-1] + 3;
    }

    return result;
}

void engval_function_initialize(double* x, unsigned int n)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        x[i] = 2;
    }
}


double penalty_function(double* x, unsigned int n)
{
    double sum_1 = 0;

    for(unsigned int i = 0; i < n; ++i)
    {
        sum_1 += pow((x[i] - 1),2)/(double)100000;
    }

    double sum_2 = 0;

    for(unsigned int j = 0; j < n; ++j)
    {
        sum_2 += pow(x[j],2) - 0.25;
    }

    return sum_1 + pow(sum_2,2);
}

void penalty_function_initialize(double* x, unsigned int n)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        x[i] = i;
    }
}

double quartic_function(double* x, unsigned int n)
{
    double result = 0;
    for(unsigned int i = 0; i < n; ++i)
    {
        result += (i+1) * pow(x[i],2);
    }
    return pow(result,2);
}

void quartic_function_initialize(double* x, unsigned int n)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        x[i] = 1;
    }
}