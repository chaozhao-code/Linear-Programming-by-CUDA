//
// Created by 唐艺峰 on 2018/7/14.
//

#ifndef INC_2D_LINEAR_PROBLEM_MODELS_H       // Avoiding to include nested too deeplt
#define INC_2D_LINEAR_PROBLEM_MODELS_H

#include "floating_number_helper.h"
#include "cuda_runtime.h"
#include "cuda.h"
typedef int I;
#define I_plus 1
#define I_minus -1
#define I_0 0

typedef struct line {
    // ax + by >= c
    double param_a;
    double param_b;
    double param_c;
    double slope_value;
	//int if_delete; // if the value is 0, it means this line is deleted, otherwise opposite
	I I_value;
	double distance_with_testline;
} line;

typedef struct point {
    // (x, y)
    double pos_x;
    double pos_y;
} point;

typedef struct intersection {
	double pos_x;
	double pos_y;
	//int line_number;
};

// the functions below may not be all used

line * generate_line_from_abc(double param_a, double param_b, double param_c); // ax + by = c
line * generate_line_from_kb(double k, double b); // y = kx + b
line * generate_line_from_2points(point * p1, point * p2); //

point * generate_point_from_xy(double pos_x, double pos_y);
point * generate_intersection_point(line * line1, line * line2);

//__device__ point * generate_intersection_point_gpu(line * line1, line * line2);

double compute_slope(line * line);
int is_parallel(line * line1, line * line2);

#endif //INC_2D_LINEAR_PROBLEM_MODELS_H
