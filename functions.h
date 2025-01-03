#pragma once

#include "functions.h"

#include <math.h>
#include <stdio.h>

void set_range_to(double* x, unsigned int n, double val);

double quadratic_function(double* x, unsigned int n);

double shannos_function(double* x, unsigned int n);

void shannos_function_initialize(double* x, unsigned int n);

double generalized_rosenbrock_function(double* x, unsigned int n);

void generalized_rosenbrock_initialize(double* x, unsigned int n);

double rastrigin_function(double* x, unsigned int n);

void rastrigin_function_initialize(double* x, unsigned int n);

double engval_function(double* x, unsigned int n);

void engval_function_initialize(double* x, unsigned int n);

double penalty_function(double* x, unsigned int n);

void penalty_function_initialize(double* x, unsigned int n);

double quartic_function(double* x, unsigned int n);

void quartic_function_initialize(double* x, unsigned int n);